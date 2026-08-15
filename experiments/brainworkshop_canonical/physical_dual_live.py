"""Human-parity Dual N-Back on public pixels plus ScreenCaptureKit PCM.

Position remains the visible play-field crop. Audio is the window's public
waveform, not a letter ID. Missing or silent Dual audio fails closed. Packed
actions use the two public keys: A for position and L for sound.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch

from neural_computer import (
    AmodalEvent,
    AmodalEventCollection,
    CapturedScreenFrame,
    CognitiveTickRuntime,
    DiscreteKeyChordOutput,
    HumanParityLiveDevice,
    MacOSApplicationWindow,
    MacOSVirtualKeyOutput,
    NativeMacOSWindowAVCapture,
    NormalizedRegion,
    ProjectedScreenPulseFrontend,
    PublicWaveformEncoder,
    VisibleColorOutcomeReader,
    pcm_rms,
)

from .controller_pretraining import (
    build_recursive_temporal_program_machine,
    load_temporal_controller_artifact,
)
from .physical_live import (
    compile_macos_av_capture_helper,
    compile_macos_keypress_helper,
)
from .rendered_live import SourcePreservingTemporalMachine

PHYSICAL_DUAL_SCHEMA = "neural-computer.brainworkshop-physical-dual-live.v1"
POSITION_KEY_CODE = 0
SOUND_KEY_CODE = 37
SILENCE_RMS = 0.008


@dataclass(frozen=True)
class PhysicalDualBrainWorkshopConfig:
    application: str = "Python"
    title_contains: str = "Brain Workshop"
    event_width: int = 16
    source_key_width: int = 4
    image_size: int = 36
    tick_hz: float = 6.0
    action_delay_seconds: float = 0.0
    capture_helper: Path | None = None
    keypress_helper: Path | None = None
    evidence_directory: Path | None = None
    stimulus_region: NormalizedRegion = field(
        default_factory=lambda: NormalizedRegion(0.26, 0.17, 0.74, 0.82)
    )
    feedback_region: NormalizedRegion = field(
        default_factory=lambda: NormalizedRegion(0.0, 0.88, 1.0, 1.0)
    )
    pulse_threshold: float = 0.008
    pulse_refractory_seconds: float = 0.5
    silence_rms: float = SILENCE_RMS
    feedback_tolerance: int = 28
    feedback_minimum_pixels: int = 12
    feedback_minimum_frames: int = 2
    schema: str = PHYSICAL_DUAL_SCHEMA

    def validate(self) -> PhysicalDualBrainWorkshopConfig:
        if self.schema != PHYSICAL_DUAL_SCHEMA:
            raise ValueError(f"unsupported physical Dual schema: {self.schema}")
        if min(self.event_width, self.source_key_width, self.image_size) < 1:
            raise ValueError("physical Dual dimensions must be positive")
        if self.tick_hz <= 0.0 or self.action_delay_seconds < 0.0:
            raise ValueError("physical Dual timing is invalid")
        if self.silence_rms < 0.0:
            raise ValueError("silence threshold cannot be negative")
        if self.capture_helper is None:
            raise ValueError("physical Dual needs the ScreenCaptureKit AV helper")
        self.stimulus_region.validate()
        self.feedback_region.validate()
        return self


class DualPulseFrontend:
    """Emit one vision event and one audio event on a visible Dual onset."""

    def __init__(
        self,
        vision: ProjectedScreenPulseFrontend,
        audio: PublicWaveformEncoder,
        *,
        silence_rms: float = SILENCE_RMS,
    ) -> None:
        if vision.event_width != audio.event_width:
            raise ValueError("Dual frontends must share an event width")
        if silence_rms < 0.0:
            raise ValueError("silence threshold cannot be negative")
        self.vision = vision
        self.audio = audio
        self.event_width = vision.event_width
        self.silence_rms = float(silence_rms)
        self._held: AmodalEvent | None = None
        self.emitted_payloads: list[torch.Tensor] = []
        self.audio_payloads: list[torch.Tensor] = []

    def _audio_ready(self, frame: CapturedScreenFrame) -> bool:
        if not frame.audio_stream_active:
            raise RuntimeError("desktop Dual has no ScreenCaptureKit audio tap")
        if frame.audio_pcm is None or frame.audio_sample_width is None:
            raise RuntimeError("desktop Dual stimulus is missing public PCM")
        return (
            pcm_rms(frame.audio_pcm, sample_width=frame.audio_sample_width)
            >= self.silence_rms
        )

    def _emit(
        self, vision_event: AmodalEvent, frame: CapturedScreenFrame
    ) -> AmodalEventCollection:
        audio_event = self.audio.encode(frame)
        self.emitted_payloads.append(vision_event.payload[0].detach().cpu().clone())
        self.audio_payloads.append(audio_event.payload[0].detach().cpu().clone())
        return AmodalEventCollection.from_events(
            (vision_event, audio_event), width=self.event_width
        )

    def observe(self, frame: CapturedScreenFrame) -> tuple[AmodalEventCollection, bool]:
        if self._held is not None:
            if not self._audio_ready(frame):
                raise RuntimeError(
                    "desktop Dual onset has no audible public PCM after one extra tick"
                )
            events = self._emit(self._held, frame)
            self._held = None
            return events, True
        vision_events, boundary = self.vision.observe(frame)
        if not boundary:
            return AmodalEventCollection.empty(1, self.event_width), False
        vision_event = AmodalEvent(
            payload=vision_events.payload[:, 0],
            source_key=None
            if vision_events.source_key is None
            else vision_events.source_key[:, 0],
            timestamp=vision_events.timestamp[:, 0]
            if vision_events.timestamp is not None
            else None,
            confidence=vision_events.confidence[:, 0],
        ).validate()
        if self._audio_ready(frame):
            return self._emit(vision_event, frame), True
        self._held = vision_event
        return AmodalEventCollection.empty(1, self.event_width), False


def dual_key_chords() -> dict[int, tuple[int, ...]]:
    return {
        0: (),
        1: (POSITION_KEY_CODE,),
        2: (SOUND_KEY_CODE,),
        3: (POSITION_KEY_CODE, SOUND_KEY_CODE),
    }


def build_physical_dual_runtime(
    machine: SourcePreservingTemporalMachine,
    config: PhysicalDualBrainWorkshopConfig,
    *,
    seed: int = 17,
) -> tuple[Any, MacOSApplicationWindow, NativeMacOSWindowAVCapture]:
    """Compose Dual I/O: pixels, PCM, packed A/L keys, visible feedback."""

    config.validate()
    if machine.event_width != config.event_width or machine.action_count != 4:
        raise ValueError("physical Dual needs event width 16 and four packed actions")
    if not getattr(machine, "pack_source_actions", False):
        raise ValueError("physical Dual requires packed two-source actions")
    machine.action_delay_seconds = config.action_delay_seconds
    window = MacOSApplicationWindow(
        config.application,
        title_contains=config.title_contains,
        require_frontmost=True,
        state_refresh_seconds=1.0,
    )
    vision = ProjectedScreenPulseFrontend(
        config.event_width,
        region=config.stimulus_region,
        source_key_width=config.source_key_width,
        image_size=config.image_size,
        change_threshold=config.pulse_threshold,
        refractory_seconds=config.pulse_refractory_seconds,
        seed=seed,
    )
    audio = PublicWaveformEncoder(
        config.event_width,
        source_key_width=config.source_key_width,
        seed=seed + 3_017,
    )
    frontend = DualPulseFrontend(vision, audio, silence_rms=config.silence_rms)
    bind = getattr(machine, "bind_executable_sources", None)
    if bind is None:
        raise ValueError("physical Dual needs bindable executable sources")
    bind(
        (
            vision.source_key.detach().reshape(1, -1),
            audio.source_key.detach().reshape(1, -1),
        )
    )
    if config.capture_helper is None:
        raise ValueError("physical Dual needs the ScreenCaptureKit AV helper")
    capture = NativeMacOSWindowAVCapture(
        window, config.capture_helper, require_audio=True
    )
    outcome_reader = VisibleColorOutcomeReader(
        region=config.feedback_region,
        negative_colors=((255, 64, 64), (64, 64, 255)),
        positive_colors=((0, 255, 0), (64, 255, 64)),
        neutral_is_positive=False,
        tolerance=config.feedback_tolerance,
        minimum_pixels=config.feedback_minimum_pixels,
        minimum_frames=config.feedback_minimum_frames,
        archive_directory=config.evidence_directory,
    )
    output = (
        MacOSVirtualKeyOutput(window, config.keypress_helper, dual_key_chords())
        if config.keypress_helper is not None
        else DiscreteKeyChordOutput(
            window, {0: (), 1: ("a",), 2: ("l",), 3: ("a", "l")}
        )
    )
    device = HumanParityLiveDevice(capture, frontend, outcome_reader, output)
    runtime = CognitiveTickRuntime(
        device,
        machine,
        {"keypress": device},
        max_tick_seconds=1.0 / config.tick_hz,
    )
    return runtime, window, capture


def run_physical_dual_loopback_probe(
    capture: NativeMacOSWindowAVCapture,
    *,
    ticks: int,
    tick_hz: float,
) -> dict[str, Any]:
    """Capture public Dual I/O without learning. Sub-minute calibration."""

    if ticks < 2 or tick_hz <= 0.0:
        raise ValueError("loopback probe settings are invalid")
    period = 1.0 / tick_hz
    frames: list[CapturedScreenFrame] = []
    started = time.monotonic()
    try:
        for index in range(ticks):
            now = time.monotonic()
            frames.append(capture.capture(now))
            remaining = started + (index + 1) * period - time.monotonic()
            if remaining > 0.0:
                time.sleep(remaining)
    finally:
        capture.close()
    energies = [
        pcm_rms(frame.audio_pcm or b"", sample_width=frame.audio_sample_width or 2)
        for frame in frames
    ]
    audible = [index for index, energy in enumerate(energies) if energy >= SILENCE_RMS]
    pairwise = []
    encoder = PublicWaveformEncoder(16, seed=17)
    if len(audible) >= 2:
        first = encoder.encode(frames[audible[0]]).payload.detach()
        second = encoder.encode(frames[audible[-1]]).payload.detach()
        pairwise.append(float(torch.linalg.vector_norm(first - second)))
    return {
        "schema": PHYSICAL_DUAL_SCHEMA,
        "ticks": ticks,
        "audio_stream_active": all(frame.audio_stream_active for frame in frames),
        "pcm_bytes": [0 if frame.audio_pcm is None else len(frame.audio_pcm) for frame in frames],
        "rms": energies,
        "audible_ticks": audible,
        "event_distance_first_last_audible": pairwise[0] if pairwise else None,
        "wall_seconds": time.monotonic() - started,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    repository = Path(__file__).parents[2]
    parser.add_argument("--seconds", type=float, default=8.0)
    parser.add_argument("--tick-hz", type=float, default=6.0)
    parser.add_argument(
        "--capture-helper",
        type=Path,
        default=Path("/tmp/neural-computer-macos-av-capture"),
    )
    parser.add_argument(
        "--keypress-helper",
        type=Path,
        default=Path("/tmp/neural-computer-macos-keypress"),
    )
    parser.add_argument("--report-out", type=Path, default=None)
    parser.add_argument(
        "--controller-artifact",
        type=Path,
        default=(
            repository
            / "artifacts/checkpoints/temporal_controller_previous_event_seed1001.pt"
        ),
    )
    arguments = parser.parse_args()
    helper = compile_macos_av_capture_helper(arguments.capture_helper)
    compile_macos_keypress_helper(arguments.keypress_helper)
    window = MacOSApplicationWindow(
        "Python", title_contains="Brain Workshop", require_frontmost=True
    )
    capture = NativeMacOSWindowAVCapture(window, helper, require_audio=True)
    ticks = max(2, math.ceil(arguments.seconds * arguments.tick_hz))
    report = run_physical_dual_loopback_probe(
        capture, ticks=ticks, tick_hz=arguments.tick_hz
    )
    report["controller_digest"] = build_recursive_temporal_program_machine(
        load_temporal_controller_artifact(arguments.controller_artifact),
        sample=False,
        max_sources=2,
        pack_source_actions=True,
    ).controller_digest()
    text = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if arguments.report_out is None:
        print(text, end="")
    else:
        arguments.report_out.parent.mkdir(parents=True, exist_ok=True)
        arguments.report_out.write_text(text)
    if not report["audio_stream_active"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
