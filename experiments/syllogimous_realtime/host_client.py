"""Client for the native ``syllogimous_host`` process.

The client deliberately exposes only decoded RGB/PCM and timestamps to the
model callback.  It does not import evaluator types or inspect game state.
"""
from __future__ import annotations

from dataclasses import dataclass
import struct
import subprocess
from time import monotonic_ns
from typing import BinaryIO, Callable, Iterable

import numpy as np

MAGIC = b"SYLLAV1\n"
HEADER = struct.Struct("<8s q I I I I I")
ALLOWED_ACTIONS = {"WAIT", "NEXT", "PREVIOUS", "TRUE", "FALSE"}

@dataclass(frozen=True)
class HostPacket:
    timestamp_ms: int
    width: int
    height: int
    frame: np.ndarray
    pcm: np.ndarray
    sample_rate: int

@dataclass(frozen=True)
class TransportEvent:
    kind: str
    wall_ns: int
    stream_timestamp_ms: int
    action: str | None = None
    modalities: tuple[str, ...] = ()

def _read_exact(stream: BinaryIO, size: int) -> bytes:
    result = bytearray()
    while len(result) < size:
        chunk = stream.read(size - len(result))
        if not chunk:
            raise EOFError(f"host stream ended after {len(result)}/{size} bytes")
        result.extend(chunk)
    return bytes(result)

def read_packet(stream: BinaryIO) -> HostPacket:
    raw = _read_exact(stream, HEADER.size)
    magic, timestamp, width, height, rgb_bytes, pcm_count, sample_rate = HEADER.unpack(raw)
    if magic != MAGIC:
        raise ValueError(f"unexpected host magic: {magic!r}")
    if width <= 0 or height <= 0 or rgb_bytes != width * height * 3:
        raise ValueError("invalid RGB dimensions in host envelope")
    rgb = np.frombuffer(_read_exact(stream, rgb_bytes), dtype=np.uint8).reshape(height, width, 3).copy()
    pcm = np.frombuffer(_read_exact(stream, pcm_count * 2), dtype="<i2").copy()
    return HostPacket(timestamp, width, height, rgb, pcm, sample_rate)

def normalize_action(text: str) -> str:
    """Convert model text to the strict action vocabulary.

    Unknown or multi-line output is intentionally reduced to WAIT; no hidden
    evaluator signal is consulted to repair an action.
    """
    token = text.strip().splitlines()[0].strip().upper() if text.strip() else "WAIT"
    return token if token in ALLOWED_ACTIONS else "WAIT"

def run_host(command: list[str], emit_action: Callable[[HostPacket], str], *,
             max_packets: int | None = None, packet_gate: Callable[[HostPacket], bool] | None = None) -> list[TransportEvent]:
    """Run a host subprocess with a causal packet->text-action loop."""
    process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                               stderr=subprocess.PIPE)
    assert process.stdin is not None and process.stdout is not None
    events: list[TransportEvent] = []
    try:
        packets = 0
        while max_packets is None or packets < max_packets:
            received_ns = monotonic_ns()
            try:
                packet = read_packet(process.stdout)
            except EOFError:
                break
            # RGB and PCM are read as one synchronized envelope.  Keep one
            # causal boundary event, but label both modalities explicitly so
            # downstream telemetry can report frame/audio timestamps separately.
            events.append(TransportEvent("stream_received", received_ns, packet.timestamp_ms,
                                         modalities=("frame", "audio")))
            if packet_gate is not None and not packet_gate(packet):
                events.append(TransportEvent("stream_suppressed", monotonic_ns(), packet.timestamp_ms))
                packets += 1
                continue
            inference_start = monotonic_ns()
            text = emit_action(packet)
            inference_end = monotonic_ns()
            events.append(TransportEvent("inference_complete", inference_end, packet.timestamp_ms))
            action = normalize_action(text)
            process.stdin.write((action + "\n").encode("ascii"))
            process.stdin.flush()
            events.append(TransportEvent("action_sent", monotonic_ns(), packet.timestamp_ms, action))
            packets += 1
    finally:
        try:
            process.stdin.close()
        except BrokenPipeError:
            pass
        process.terminate()
        process.wait(timeout=5)
    return events
