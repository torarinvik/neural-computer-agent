"""Independent, versioned serialization for the production runtime."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Any

import torch

from .interface import EVENT_SCHEMA, INTENTION_SCHEMA
from .retention import CapabilityRetentionLedger
from .runtime import RUNTIME_FORMAT, AmodalControllerRuntime

LEGACY_RUNTIME_FORMAT = "neural-computer.amodal-runtime.v1"
LEGACY_RUNTIME_FORMAT_V2 = "neural-computer.amodal-runtime.v2"
LEGACY_RUNTIME_FORMAT_V3 = "neural-computer.amodal-runtime.v3"
LEGACY_RUNTIME_FORMAT_V4 = "neural-computer.amodal-runtime.v4"
LEGACY_RUNTIME_FORMAT_V5 = "neural-computer.amodal-runtime.v5"
LEGACY_RUNTIME_FORMAT_V6 = "neural-computer.amodal-runtime.v6"
LEGACY_RUNTIME_FORMAT_V7 = "neural-computer.amodal-runtime.v7"
LEGACY_RUNTIME_FORMAT_V8 = "neural-computer.amodal-runtime.v8"
LEGACY_RUNTIME_FORMAT_V9 = "neural-computer.amodal-runtime.v9"
LEGACY_RUNTIME_FORMAT_V10 = "neural-computer.amodal-runtime.v10"
LEGACY_RUNTIME_FORMAT_V11 = "neural-computer.amodal-runtime.v11"
LEGACY_RUNTIME_FORMAT_V12 = "neural-computer.amodal-runtime.v12"
LEGACY_RUNTIME_FORMAT_V13 = "neural-computer.amodal-runtime.v13"
LEGACY_RUNTIME_FORMAT_V14 = "neural-computer.amodal-runtime.v14"
LEGACY_RUNTIME_FORMAT_V15 = "neural-computer.amodal-runtime.v15"
LEGACY_RUNTIME_FORMAT_V16 = "neural-computer.amodal-runtime.v16"
LEGACY_RUNTIME_FORMAT_V17 = "neural-computer.amodal-runtime.v17"
LEGACY_RUNTIME_FORMAT_V18 = "neural-computer.amodal-runtime.v18"
LEGACY_RUNTIME_FORMAT_V19 = "neural-computer.amodal-runtime.v19"
LEGACY_RUNTIME_FORMAT_V20 = "neural-computer.amodal-runtime.v20"
LEGACY_RUNTIME_FORMAT_V21 = "neural-computer.amodal-runtime.v21"
LEGACY_RUNTIME_FORMAT_V22 = "neural-computer.amodal-runtime.v22"
LEGACY_RUNTIME_FORMAT_V23 = "neural-computer.amodal-runtime.v23"
LEGACY_RUNTIME_FORMAT_V24 = "neural-computer.amodal-runtime.v24"
LEGACY_RUNTIME_FORMAT_V25 = "neural-computer.amodal-runtime.v25"
LEGACY_RUNTIME_FORMAT_V26 = "neural-computer.amodal-runtime.v26"
LEGACY_RUNTIME_FORMAT_V27 = "neural-computer.amodal-runtime.v27"
LEGACY_RUNTIME_FORMAT_V28 = "neural-computer.amodal-runtime.v28"
LEGACY_RUNTIME_FORMAT_V29 = "neural-computer.amodal-runtime.v29"


def _memory_configuration_matches(
    expected: object, payload: object
) -> bool:
    if expected is None or payload is None:
        return expected is payload
    if not isinstance(expected, dict) or not isinstance(payload, dict):
        return expected == payload
    expected_memory = dict(expected)
    payload_memory = dict(payload)
    expected_memory.pop("persistence", None)
    payload_memory.pop("persistence", None)
    if "write_match_threshold" not in payload_memory:
        expected_memory.pop("write_match_threshold", None)
    if "scope_capacity" not in payload_memory and expected_memory.get("scope_capacity") == 1:
        expected_memory.pop("scope_capacity", None)
    return expected_memory == payload_memory


def _runtime_configuration_matches(
    expected: dict[str, object],
    payload: object,
    *,
    allow_legacy_representation: bool = False,
) -> bool:
    if not isinstance(payload, dict):
        return False
    if set(expected) != set(payload):
        optional = {
            "representation_space_schema",
            "event_space_id",
            "state_space_id",
            "intention_space_id",
        }
        if not allow_legacy_representation or set(payload) != set(expected) - optional:
            return False
        expected = {key: value for key, value in expected.items() if key not in optional}
    return all(
        _memory_configuration_matches(expected[key], payload[key])
        if key == "memory"
        else expected[key] == payload[key]
        for key in expected
    )


def _atomic_torch_save(payload: object, path: Path) -> None:
    """Write a checkpoint without leaving a partially written target."""
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    os.close(descriptor)
    temporary_path = Path(temporary_name)
    try:
        torch.save(payload, temporary_path)
        temporary_path.replace(path)
    finally:
        temporary_path.unlink(missing_ok=True)


def save_runtime(
    runtime: AmodalControllerRuntime,
    path: Path,
    *,
    provenance: dict[str, object] | None = None,
) -> None:
    """Save independently loadable controller, adapter, and memory components."""
    path.parent.mkdir(parents=True, exist_ok=True)
    _atomic_torch_save(runtime.checkpoint_payload(provenance=provenance), path)


def load_runtime_components(
    runtime: AmodalControllerRuntime,
    path: Path,
    *,
    map_location: torch.device | str = "cpu",
) -> dict[str, Any]:
    """Load a canonical payload into an already-constructed runtime.

    The caller constructs the encoder/decoder set explicitly.  This makes
    adapter replacement visible and prevents checkpoint metadata from creating
    hidden modality branches.
    """
    payload = torch.load(path, map_location=map_location, weights_only=False)
    payload_format = payload.get("format")
    if payload_format not in {
        RUNTIME_FORMAT,
        LEGACY_RUNTIME_FORMAT_V29,
        LEGACY_RUNTIME_FORMAT_V28,
        LEGACY_RUNTIME_FORMAT_V27,
        LEGACY_RUNTIME_FORMAT_V26,
        LEGACY_RUNTIME_FORMAT_V25,
        LEGACY_RUNTIME_FORMAT_V24,
        LEGACY_RUNTIME_FORMAT_V23,
        LEGACY_RUNTIME_FORMAT,
        LEGACY_RUNTIME_FORMAT_V2,
        LEGACY_RUNTIME_FORMAT_V3,
        LEGACY_RUNTIME_FORMAT_V4,
        LEGACY_RUNTIME_FORMAT_V5,
        LEGACY_RUNTIME_FORMAT_V6,
        LEGACY_RUNTIME_FORMAT_V7,
        LEGACY_RUNTIME_FORMAT_V8,
        LEGACY_RUNTIME_FORMAT_V9,
        LEGACY_RUNTIME_FORMAT_V10,
        LEGACY_RUNTIME_FORMAT_V11,
        LEGACY_RUNTIME_FORMAT_V12,
        LEGACY_RUNTIME_FORMAT_V13,
        LEGACY_RUNTIME_FORMAT_V14,
        LEGACY_RUNTIME_FORMAT_V15,
        LEGACY_RUNTIME_FORMAT_V16,
        LEGACY_RUNTIME_FORMAT_V17,
        LEGACY_RUNTIME_FORMAT_V22,
        LEGACY_RUNTIME_FORMAT_V21,
        LEGACY_RUNTIME_FORMAT_V20,
        LEGACY_RUNTIME_FORMAT_V19,
        LEGACY_RUNTIME_FORMAT_V18,
    }:
        raise ValueError("unsupported amodal runtime checkpoint format")
    if payload.get("event_schema") != EVENT_SCHEMA or payload.get("intention_schema") != INTENTION_SCHEMA:
        raise ValueError("unsupported neural-IR schema in runtime checkpoint")
    expected_configuration = runtime.configuration()
    legacy = payload_format in {
        LEGACY_RUNTIME_FORMAT_V28,
        LEGACY_RUNTIME_FORMAT_V27,
        LEGACY_RUNTIME_FORMAT_V26,
        LEGACY_RUNTIME_FORMAT_V25,
        LEGACY_RUNTIME_FORMAT,
        LEGACY_RUNTIME_FORMAT_V2,
        LEGACY_RUNTIME_FORMAT_V3,
        LEGACY_RUNTIME_FORMAT_V4,
        LEGACY_RUNTIME_FORMAT_V5,
        LEGACY_RUNTIME_FORMAT_V6,
        LEGACY_RUNTIME_FORMAT_V7,
        LEGACY_RUNTIME_FORMAT_V8,
        LEGACY_RUNTIME_FORMAT_V9,
        LEGACY_RUNTIME_FORMAT_V10,
        LEGACY_RUNTIME_FORMAT_V11,
        LEGACY_RUNTIME_FORMAT_V12,
        LEGACY_RUNTIME_FORMAT_V13,
        LEGACY_RUNTIME_FORMAT_V14,
        LEGACY_RUNTIME_FORMAT_V15,
        LEGACY_RUNTIME_FORMAT_V16,
        LEGACY_RUNTIME_FORMAT_V17,
        LEGACY_RUNTIME_FORMAT_V22,
        LEGACY_RUNTIME_FORMAT_V21,
        LEGACY_RUNTIME_FORMAT_V20,
        LEGACY_RUNTIME_FORMAT_V19,
        LEGACY_RUNTIME_FORMAT_V18,
        LEGACY_RUNTIME_FORMAT_V23,
        LEGACY_RUNTIME_FORMAT_V24,
        LEGACY_RUNTIME_FORMAT_V26,
        LEGACY_RUNTIME_FORMAT_V25,
    }
    if legacy:
        legacy_controller = dict(expected_configuration["controller"])
        if "source_credit_projection" not in payload["configuration"]["controller"]:
            # This field was introduced with v15; accept genuinely old
            # payloads while keeping exact configuration checks for fields
            # that existed in their declared schema.
            legacy_controller.pop("source_credit_projection", None)
        elif payload_format in {
            LEGACY_RUNTIME_FORMAT_V15,
            LEGACY_RUNTIME_FORMAT_V16,
            LEGACY_RUNTIME_FORMAT_V17,
            LEGACY_RUNTIME_FORMAT_V23,
            LEGACY_RUNTIME_FORMAT_V24,
            LEGACY_RUNTIME_FORMAT_V26,
            LEGACY_RUNTIME_FORMAT_V25,
            LEGACY_RUNTIME_FORMAT_V28,
            LEGACY_RUNTIME_FORMAT_V27,
            LEGACY_RUNTIME_FORMAT_V22,
            LEGACY_RUNTIME_FORMAT_V21,
            LEGACY_RUNTIME_FORMAT_V20,
            LEGACY_RUNTIME_FORMAT_V19,
            LEGACY_RUNTIME_FORMAT_V18,
        }:
            legacy_controller["source_credit_projection"] = payload["configuration"]["controller"][
                "source_credit_projection"
            ]
        if payload_format == LEGACY_RUNTIME_FORMAT_V13:
            legacy_controller["source_trust_binding"] = bool(
                expected_configuration["controller"]["source_key_width"]
            )
            legacy_controller["source_trust_binding_scale"] = 0.5
        elif payload_format == LEGACY_RUNTIME_FORMAT_V14:
            legacy_controller["source_trust_binding"] = bool(
                expected_configuration["controller"]["source_key_width"]
            )
            legacy_controller["source_trust_binding_scale"] = 0.25
        elif payload_format in {
            LEGACY_RUNTIME_FORMAT_V15,
            LEGACY_RUNTIME_FORMAT_V16,
            LEGACY_RUNTIME_FORMAT_V17,
            LEGACY_RUNTIME_FORMAT_V23,
            LEGACY_RUNTIME_FORMAT_V24,
            LEGACY_RUNTIME_FORMAT_V26,
            LEGACY_RUNTIME_FORMAT_V25,
            LEGACY_RUNTIME_FORMAT_V28,
            LEGACY_RUNTIME_FORMAT_V27,
            LEGACY_RUNTIME_FORMAT_V22,
            LEGACY_RUNTIME_FORMAT_V21,
            LEGACY_RUNTIME_FORMAT_V20,
            LEGACY_RUNTIME_FORMAT_V19,
            LEGACY_RUNTIME_FORMAT_V18,
        }:
            legacy_controller["source_trust_binding"] = payload["configuration"][
                "controller"
            ].get("source_trust_binding", False)
            legacy_controller["source_trust_binding_scale"] = payload[
                "configuration"
            ]["controller"].get("source_trust_binding_scale", 0.25)
        else:
            legacy_controller.pop("source_trust_binding", None)
            legacy_controller.pop("source_trust_binding_scale", None)
        if payload_format not in {
            LEGACY_RUNTIME_FORMAT_V12,
            LEGACY_RUNTIME_FORMAT_V13,
            LEGACY_RUNTIME_FORMAT_V14,
            LEGACY_RUNTIME_FORMAT_V15,
            LEGACY_RUNTIME_FORMAT_V16,
            LEGACY_RUNTIME_FORMAT_V17,
            LEGACY_RUNTIME_FORMAT_V23,
            LEGACY_RUNTIME_FORMAT_V24,
            LEGACY_RUNTIME_FORMAT_V26,
            LEGACY_RUNTIME_FORMAT_V25,
            LEGACY_RUNTIME_FORMAT_V28,
            LEGACY_RUNTIME_FORMAT_V27,
            LEGACY_RUNTIME_FORMAT_V22,
            LEGACY_RUNTIME_FORMAT_V21,
            LEGACY_RUNTIME_FORMAT_V20,
            LEGACY_RUNTIME_FORMAT_V19,
            LEGACY_RUNTIME_FORMAT_V18,
        }:
            legacy_controller.pop("source_credit_policy", None)
            legacy_controller.pop("source_credit_hidden", None)
        if payload_format not in {
            LEGACY_RUNTIME_FORMAT_V10,
            LEGACY_RUNTIME_FORMAT_V11,
            LEGACY_RUNTIME_FORMAT_V12,
            LEGACY_RUNTIME_FORMAT_V13,
            LEGACY_RUNTIME_FORMAT_V14,
            LEGACY_RUNTIME_FORMAT_V15,
            LEGACY_RUNTIME_FORMAT_V16,
            LEGACY_RUNTIME_FORMAT_V17,
            LEGACY_RUNTIME_FORMAT_V23,
            LEGACY_RUNTIME_FORMAT_V24,
            LEGACY_RUNTIME_FORMAT_V26,
            LEGACY_RUNTIME_FORMAT_V25,
            LEGACY_RUNTIME_FORMAT_V28,
            LEGACY_RUNTIME_FORMAT_V27,
            LEGACY_RUNTIME_FORMAT_V22,
            LEGACY_RUNTIME_FORMAT_V21,
            LEGACY_RUNTIME_FORMAT_V20,
            LEGACY_RUNTIME_FORMAT_V19,
            LEGACY_RUNTIME_FORMAT_V18,
        }:
            legacy_controller.pop("event_feedback_source_relevance", None)
        if payload_format not in {
            LEGACY_RUNTIME_FORMAT_V11,
            LEGACY_RUNTIME_FORMAT_V12,
            LEGACY_RUNTIME_FORMAT_V13,
            LEGACY_RUNTIME_FORMAT_V14,
            LEGACY_RUNTIME_FORMAT_V15,
            LEGACY_RUNTIME_FORMAT_V16,
            LEGACY_RUNTIME_FORMAT_V17,
            LEGACY_RUNTIME_FORMAT_V23,
            LEGACY_RUNTIME_FORMAT_V24,
            LEGACY_RUNTIME_FORMAT_V26,
            LEGACY_RUNTIME_FORMAT_V25,
            LEGACY_RUNTIME_FORMAT_V28,
            LEGACY_RUNTIME_FORMAT_V27,
            LEGACY_RUNTIME_FORMAT_V22,
            LEGACY_RUNTIME_FORMAT_V21,
            LEGACY_RUNTIME_FORMAT_V20,
            LEGACY_RUNTIME_FORMAT_V19,
            LEGACY_RUNTIME_FORMAT_V18,
        }:
            legacy_controller.pop("source_credit_state", None)
            legacy_controller.pop("source_credit_decay", None)
        if payload_format == LEGACY_RUNTIME_FORMAT:
            legacy_controller.pop("execution_hidden", None)
            legacy_controller.pop("execution_states", None)
            legacy_controller.pop("execution_transport_features", None)
            legacy_controller.pop("execution_timeout_policy", None)
            legacy_controller.pop("event_pair_attention", None)
            legacy_controller.pop("event_pair_relevance", None)
            legacy_controller["schema"] = "neural-computer.controller.v1"
        elif payload_format == LEGACY_RUNTIME_FORMAT_V2:
            legacy_controller["schema"] = "neural-computer.controller.v2"
            legacy_controller.pop("execution_transport_features", None)
            legacy_controller.pop("execution_timeout_policy", None)
            legacy_controller.pop("event_pair_attention", None)
            legacy_controller.pop("event_pair_relevance", None)
        elif payload_format == LEGACY_RUNTIME_FORMAT_V3:
            legacy_controller["schema"] = "neural-computer.controller.v3"
            legacy_controller.pop("execution_transport_features", None)
            legacy_controller.pop("execution_timeout_policy", None)
            legacy_controller.pop("event_pair_attention", None)
            legacy_controller.pop("event_pair_relevance", None)
        elif payload_format == LEGACY_RUNTIME_FORMAT_V4:
            legacy_controller["schema"] = "neural-computer.controller.v4"
            legacy_controller["execution_transport_features"] = 3
            legacy_controller.pop("execution_timeout_policy", None)
            legacy_controller.pop("event_pair_attention", None)
            legacy_controller.pop("event_pair_relevance", None)
        elif payload_format == LEGACY_RUNTIME_FORMAT_V5:
            legacy_controller["schema"] = "neural-computer.controller.v5"
            legacy_controller["execution_transport_features"] = 4
            legacy_controller.pop("execution_timeout_policy", None)
            legacy_controller.pop("event_pair_attention", None)
            legacy_controller.pop("event_pair_relevance", None)
        elif payload_format == LEGACY_RUNTIME_FORMAT_V6:
            legacy_controller["schema"] = "neural-computer.controller.v6"
            legacy_controller.pop("event_pair_attention", None)
            legacy_controller.pop("event_pair_relevance", None)
        elif payload_format == LEGACY_RUNTIME_FORMAT_V7:
            legacy_controller["schema"] = "neural-computer.controller.v7"
            legacy_controller.pop("event_pair_relevance", None)
        elif payload_format == LEGACY_RUNTIME_FORMAT_V8:
            legacy_controller["schema"] = "neural-computer.controller.v8"
        elif payload_format == LEGACY_RUNTIME_FORMAT_V9:
            legacy_controller["schema"] = "neural-computer.controller.v9"
        elif payload_format == LEGACY_RUNTIME_FORMAT_V10:
            legacy_controller["schema"] = "neural-computer.controller.v10"
        elif payload_format == LEGACY_RUNTIME_FORMAT_V11:
            legacy_controller["schema"] = "neural-computer.controller.v11"
        elif payload_format == LEGACY_RUNTIME_FORMAT_V12:
            legacy_controller["schema"] = "neural-computer.controller.v12"
        elif payload_format == LEGACY_RUNTIME_FORMAT_V13:
            legacy_controller["schema"] = "neural-computer.controller.v13"
        elif payload_format == LEGACY_RUNTIME_FORMAT_V14:
            legacy_controller["schema"] = "neural-computer.controller.v14"
        elif payload_format == LEGACY_RUNTIME_FORMAT_V15:
            legacy_controller["schema"] = "neural-computer.controller.v15"
        elif payload_format == LEGACY_RUNTIME_FORMAT_V16:
            legacy_controller["schema"] = "neural-computer.controller.v16"
        elif payload_format == LEGACY_RUNTIME_FORMAT_V17:
            legacy_controller["schema"] = "neural-computer.controller.v17"
        elif payload_format == LEGACY_RUNTIME_FORMAT_V18:
            legacy_controller["schema"] = "neural-computer.controller.v18"
        elif payload_format == LEGACY_RUNTIME_FORMAT_V19:
            legacy_controller["schema"] = "neural-computer.controller.v19"
        elif payload_format == LEGACY_RUNTIME_FORMAT_V20:
            legacy_controller["schema"] = "neural-computer.controller.v20"
        elif payload_format == LEGACY_RUNTIME_FORMAT_V21:
            legacy_controller["schema"] = "neural-computer.controller.v21"
        elif payload_format == LEGACY_RUNTIME_FORMAT_V22:
            legacy_controller["schema"] = "neural-computer.controller.v22"
        elif payload_format == LEGACY_RUNTIME_FORMAT_V23:
            legacy_controller["schema"] = "neural-computer.controller.v23"
        elif payload_format == LEGACY_RUNTIME_FORMAT_V24:
            legacy_controller["schema"] = "neural-computer.controller.v24"
        elif payload_format == LEGACY_RUNTIME_FORMAT_V25:
            legacy_controller["schema"] = "neural-computer.controller.v25"
        elif payload_format == LEGACY_RUNTIME_FORMAT_V26:
            legacy_controller["schema"] = "neural-computer.controller.v26"
        elif payload_format == LEGACY_RUNTIME_FORMAT_V28:
            legacy_controller["schema"] = "neural-computer.controller.v28"
        payload_controller = payload["configuration"]["controller"]
        if payload_format == LEGACY_RUNTIME_FORMAT_V27:
            for growth_field in (
                "growth_register_widths",
                "growth_prior_only_from",
                "growth_recurrent_from",
                "growth_boundary",
            ):
                legacy_controller.pop(growth_field, None)
            legacy_controller["schema"] = "neural-computer.controller.v27"
        for compatibility_field in (
            "memory_address",
            "event_address_relevance",
            "memory_write_policy",
            "memory_write_hidden",
            "memory_write_sampling",
            "memory_write_event_window",
            "memory_write_event_match",
            "memory_value_feedback",
            "memory_value_stable",
        ):
            if compatibility_field not in payload_controller:
                legacy_controller.pop(compatibility_field, None)
        if payload_format in {
            LEGACY_RUNTIME_FORMAT_V24,
            LEGACY_RUNTIME_FORMAT_V26,
            LEGACY_RUNTIME_FORMAT_V25,
            LEGACY_RUNTIME_FORMAT_V23,
            LEGACY_RUNTIME_FORMAT_V20,
            LEGACY_RUNTIME_FORMAT_V22,
            LEGACY_RUNTIME_FORMAT_V21,
        }:
            legacy_controller["memory_address"] = payload_controller["memory_address"]
            legacy_controller["memory_write_event_window"] = payload_controller[
                "memory_write_event_window"
            ]
        if payload_format == LEGACY_RUNTIME_FORMAT_V25:
            legacy_controller["memory_write_event_match"] = payload_controller[
                "memory_write_event_match"
            ]
        if payload_format not in {
            LEGACY_RUNTIME_FORMAT_V9,
            LEGACY_RUNTIME_FORMAT_V10,
            LEGACY_RUNTIME_FORMAT_V11,
            LEGACY_RUNTIME_FORMAT_V12,
            LEGACY_RUNTIME_FORMAT_V13,
            LEGACY_RUNTIME_FORMAT_V14,
            LEGACY_RUNTIME_FORMAT_V15,
            LEGACY_RUNTIME_FORMAT_V16,
            LEGACY_RUNTIME_FORMAT_V17,
            LEGACY_RUNTIME_FORMAT_V23,
            LEGACY_RUNTIME_FORMAT_V24,
            LEGACY_RUNTIME_FORMAT_V26,
            LEGACY_RUNTIME_FORMAT_V25,
            LEGACY_RUNTIME_FORMAT_V28,
            LEGACY_RUNTIME_FORMAT_V27,
            LEGACY_RUNTIME_FORMAT_V22,
            LEGACY_RUNTIME_FORMAT_V21,
            LEGACY_RUNTIME_FORMAT_V20,
            LEGACY_RUNTIME_FORMAT_V19,
            LEGACY_RUNTIME_FORMAT_V18,
        }:
            legacy_controller.pop("event_feedback_relevance", None)
        if payload_format in {LEGACY_RUNTIME_FORMAT, LEGACY_RUNTIME_FORMAT_V2}:
            legacy_controller.pop("execution_transport_policy", None)
        expected_configuration = {
            **expected_configuration,
            "format": payload_format,
            "controller": legacy_controller,
        }
    if payload_format == LEGACY_RUNTIME_FORMAT_V29:
        expected_configuration = {
            **expected_configuration,
            "format": payload_format,
        }
    if not _runtime_configuration_matches(
        expected_configuration,
        payload.get("configuration"),
        allow_legacy_representation=payload_format == LEGACY_RUNTIME_FORMAT_V29,
    ):
        raise ValueError("checkpoint configuration does not match runtime")
    components = payload["components"]
    if payload_format in {
        LEGACY_RUNTIME_FORMAT_V3,
        LEGACY_RUNTIME_FORMAT_V4,
        LEGACY_RUNTIME_FORMAT_V5,
        LEGACY_RUNTIME_FORMAT_V6,
        LEGACY_RUNTIME_FORMAT_V7,
        LEGACY_RUNTIME_FORMAT_V8,
        LEGACY_RUNTIME_FORMAT_V9,
        LEGACY_RUNTIME_FORMAT_V10,
        LEGACY_RUNTIME_FORMAT_V11,
        LEGACY_RUNTIME_FORMAT_V12,
        LEGACY_RUNTIME_FORMAT_V13,
        LEGACY_RUNTIME_FORMAT_V14,
        LEGACY_RUNTIME_FORMAT_V15,
        LEGACY_RUNTIME_FORMAT_V16,
        LEGACY_RUNTIME_FORMAT_V17,
        LEGACY_RUNTIME_FORMAT_V22,
        LEGACY_RUNTIME_FORMAT_V21,
        LEGACY_RUNTIME_FORMAT_V20,
        LEGACY_RUNTIME_FORMAT_V19,
        LEGACY_RUNTIME_FORMAT_V18,
    }:
        # The execution transport head changed shape at each version. Keep
        # the current initialized compatibility prior and load every other
        # component from the old checkpoint.
        components = dict(components)
        components["controller"] = {
            key: value
            for key, value in components["controller"].items()
            if not key.startswith("execution_transport_policy.")
        }
    if payload_format in {
        LEGACY_RUNTIME_FORMAT_V14,
        LEGACY_RUNTIME_FORMAT_V15,
        LEGACY_RUNTIME_FORMAT_V16,
        LEGACY_RUNTIME_FORMAT_V17,
        LEGACY_RUNTIME_FORMAT_V22,
        LEGACY_RUNTIME_FORMAT_V21,
        LEGACY_RUNTIME_FORMAT_V20,
        LEGACY_RUNTIME_FORMAT_V19,
        LEGACY_RUNTIME_FORMAT_V18,
    }:
        # These legacy heads either had an uncalibrated vector update or used
        # a different trust update scale. v17 starts with a neutral,
        # normalized, count-invariant credit head.
        components = dict(components)
        components["controller"] = {
            key: value
            for key, value in components["controller"].items()
            if not key.startswith("source_credit_policy.")
        }
    runtime.load_component_state_dicts(components, allow_missing_execution=legacy)
    retention_payload = payload.get("retention_ledger")
    if retention_payload is not None and runtime.memory is not None:
        if not hasattr(runtime.memory, "retention"):
            raise ValueError("runtime memory does not support retention state")
        previous_retention = runtime.memory.retention
        try:
            runtime.memory.retention = CapabilityRetentionLedger.from_payload(
                retention_payload
            )
        except Exception:
            runtime.memory.retention = previous_retention
            raise
    if legacy:
        # v23 and earlier used transport-augmented event addresses. v24
        # already uses payload-only addresses; v25 adds the feedback residual.
        if payload_format not in {
            LEGACY_RUNTIME_FORMAT_V24,
            LEGACY_RUNTIME_FORMAT_V26,
            LEGACY_RUNTIME_FORMAT_V25,
            LEGACY_RUNTIME_FORMAT_V28,
            LEGACY_RUNTIME_FORMAT_V27,
        }:
            runtime.controller.stable_memory_address = False
        runtime.controller.memory_address_residual = (
            payload["configuration"]["controller"].get("memory_address")
            == "latest_event_payload_residual_v2"
        )
        runtime.controller.memory_value_feedback_enabled = (
            payload["configuration"]["controller"].get("memory_value_feedback")
            == "feedback_residual_v1"
        )
        runtime.controller.source_trust_binding_scale = (
            0.5
            if payload_format == LEGACY_RUNTIME_FORMAT_V13
            else 0.25
            if payload_format == LEGACY_RUNTIME_FORMAT_V14
            else 0.25
            if payload_format == LEGACY_RUNTIME_FORMAT_V15
            else 0.25
            if payload_format == LEGACY_RUNTIME_FORMAT_V16
            else 0.25
            if payload_format == LEGACY_RUNTIME_FORMAT_V17
            else 0.0
        )
    return payload
