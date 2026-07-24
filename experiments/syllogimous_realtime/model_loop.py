"""Causal host/model adapter.

The callback receives only decoded sensory packets and returns text.  It never
gets a Question, seed, answer, deadline, or evaluator object.
"""
from __future__ import annotations
from dataclasses import dataclass
from time import monotonic_ns
from typing import Callable, Iterable, Any
import json

@dataclass(frozen=True)
class BoundaryEvent:
    kind: str
    timestamp_ns: int
    timestamp_ms: int

def run_stream(packets: Iterable[Any], emit_action: Callable[[Any], str],
               send_action: Callable[[str], None], log_path: str | None = None) -> list[BoundaryEvent]:
    events: list[BoundaryEvent] = []
    for packet in packets:
        now = monotonic_ns()
        events.append(BoundaryEvent("frame_audio_received", now, int(packet.timestamp_ms)))
        inference_start = monotonic_ns()
        action = emit_action(packet)
        inference_end = monotonic_ns()
        events.append(BoundaryEvent("inference_complete", inference_end, int(packet.timestamp_ms)))
        send_action(action)
        events.append(BoundaryEvent("action_sent", monotonic_ns(), int(packet.timestamp_ms)))
    if log_path:
        with open(log_path, "w", encoding="utf-8") as handle:
            for event in events:
                handle.write(json.dumps({"kind": event.kind, "timestamp_ns": event.timestamp_ns,
                                         "stream_timestamp_ms": event.timestamp_ms}) + "\n")
    return events
