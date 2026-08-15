"""Human-parity physical Brain Workshop composition for macOS.

The learner sees captured window pixels only, emits ordinary keypresses, and
receives scalar outcomes derived solely from the application's visible feedback
colors.  The upstream GUI is never imported and no private game state is read.
"""

from __future__ import annotations

import json
import subprocess
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

from neural_computer import (
    CognitiveTickRuntime,
    DiscreteKeyChordOutput,
    FFmpegMacOSWindowCapture,
    HumanParityLiveDevice,
    MacOSApplicationWindow,
    MacOSVirtualKeyOutput,
    NativeMacOSWindowCapture,
    NormalizedRegion,
    ProjectedScreenPulseFrontend,
    VisibleColorOutcomeReader,
)

from .rendered_live import SourcePreservingTemporalMachine

PHYSICAL_BRAINWORKSHOP_SCHEMA = "neural-computer.brainworkshop-physical-live.v1"


@dataclass(frozen=True)
class PhysicalBrainWorkshopConfig:
    """External public-interface calibration for Position N-Back."""

    application: str = "Python"
    title_contains: str = "Brain Workshop"
    event_width: int = 16
    source_key_width: int = 4
    image_size: int = 36
    tick_hz: float = 6.0
    action_delay_seconds: float = 0.0
    capture_backend: str = "screencapture"
    screen_input_index: int = 3
    backing_scale: float = 2.0
    keypress_helper: Path | None = None
    capture_helper: Path | None = None
    evidence_directory: Path | None = None
    stimulus_region: NormalizedRegion = field(
        default_factory=lambda: NormalizedRegion(0.26, 0.17, 0.74, 0.82)
    )
    feedback_region: NormalizedRegion = field(
        default_factory=lambda: NormalizedRegion(0.0, 0.88, 1.0, 1.0)
    )
    pulse_threshold: float = 0.008
    pulse_refractory_seconds: float = 0.5
    feedback_tolerance: int = 28
    feedback_minimum_pixels: int = 12
    feedback_minimum_frames: int = 2
    schema: str = PHYSICAL_BRAINWORKSHOP_SCHEMA

    def validate(self) -> PhysicalBrainWorkshopConfig:
        if self.schema != PHYSICAL_BRAINWORKSHOP_SCHEMA:
            raise ValueError(
                f"unsupported physical Brain Workshop schema: {self.schema}"
            )
        if min(self.event_width, self.source_key_width, self.image_size) < 1:
            raise ValueError("physical Brain Workshop dimensions must be positive")
        if (
            self.tick_hz <= 0.0
            or self.action_delay_seconds < 0.0
            or self.screen_input_index < 0
            or self.backing_scale <= 0.0
        ):
            raise ValueError("physical Brain Workshop capture settings are invalid")
        if self.capture_backend not in {"native", "screencapture", "ffmpeg"}:
            raise ValueError("unsupported physical screen capture backend")
        if self.capture_backend == "native" and self.capture_helper is None:
            raise ValueError("native screen capture needs a compiled helper")
        self.stimulus_region.validate()
        self.feedback_region.validate()
        return self


@dataclass(frozen=True)
class PhysicalBrainWorkshopReport:
    ticks: int
    input_events: int
    unique_public_outcomes: int
    optimizer_updates: int
    program_file_updates: int
    emitted_actions: int
    deadline_misses: int
    elapsed_seconds: float
    tick_hz: float
    action_delay_seconds: float
    capture_backend: str
    rewards: tuple[float, ...]
    actions: tuple[int, ...]
    propensities: tuple[float, ...]
    evidence_digests: tuple[tuple[str, ...], ...]
    event_payloads: tuple[tuple[float, ...], ...]
    evidence_archive: str | None
    total_seconds_p50: float | None = None
    total_seconds_p99: float | None = None
    schema: str = PHYSICAL_BRAINWORKSHOP_SCHEMA

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


def build_physical_brainworkshop_runtime(
    machine: SourcePreservingTemporalMachine,
    config: PhysicalBrainWorkshopConfig,
    *,
    seed: int = 17,
) -> tuple[CognitiveTickRuntime, MacOSApplicationWindow, object]:
    """Compose the generic runtime for the visible Position N-Back mode.

    Action index zero withholds a response and index one presses the visible
    position-match key.  These meanings terminate at the replaceable keyboard
    decoder and never enter the controller.
    """

    config.validate()
    if machine.event_width != config.event_width or machine.action_count != 2:
        raise ValueError("Position N-Back needs matching event width and two actions")
    machine.action_delay_seconds = config.action_delay_seconds
    window = MacOSApplicationWindow(
        config.application,
        title_contains=config.title_contains,
        require_frontmost=True,
        state_refresh_seconds=1.0,
    )
    frontend = ProjectedScreenPulseFrontend(
        config.event_width,
        region=config.stimulus_region,
        source_key_width=config.source_key_width,
        image_size=config.image_size,
        change_threshold=config.pulse_threshold,
        refractory_seconds=config.pulse_refractory_seconds,
        seed=seed,
    )
    if config.capture_backend == "ffmpeg":
        capture: object = FFmpegMacOSWindowCapture(
            window,
            screen_input_index=config.screen_input_index,
            frames_per_second=config.tick_hz,
            backing_scale=config.backing_scale,
        )
    elif config.capture_backend == "screencapture":
        state = window.state()
        window.lock_bounds(state.bounds)
        capture = window
    else:
        if config.capture_helper is None:
            raise ValueError("native screen capture helper is missing")
        capture = NativeMacOSWindowCapture(window, config.capture_helper)
    outcome_reader = VisibleColorOutcomeReader(
        region=config.feedback_region,
        negative_colors=((255, 64, 64), (64, 64, 255)),
        # Brain Workshop 5.0's macOS renderer presents correct feedback as
        # saturated green even though its configuration names the lighter
        # fallback color. Accept both public pixel values; this calibration is
        # confined to the replaceable screen adapter.
        positive_colors=((0, 255, 0), (64, 255, 64)),
        neutral_is_positive=False,
        tolerance=config.feedback_tolerance,
        minimum_pixels=config.feedback_minimum_pixels,
        minimum_frames=config.feedback_minimum_frames,
        archive_directory=config.evidence_directory,
    )
    output = (
        MacOSVirtualKeyOutput(window, config.keypress_helper, {0: (), 1: (0,)})
        if config.keypress_helper is not None
        else DiscreteKeyChordOutput(window, {0: (), 1: ("a",)})
    )
    device = HumanParityLiveDevice(capture, frontend, outcome_reader, output)
    runtime = CognitiveTickRuntime(
        device,
        machine,
        {"keypress": device},
        max_tick_seconds=1.0 / config.tick_hz,
    )
    return runtime, window, capture


def compile_macos_keypress_helper(output: Path) -> Path:
    """Compile the repository's modality-agnostic native key transport."""

    source = Path(__file__).parents[2] / "tools" / "macos_keypress.c"
    if not source.is_file():
        raise FileNotFoundError(source)
    output.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "clang",
            "-O2",
            "-framework",
            "ApplicationServices",
            str(source),
            "-o",
            str(output),
        ],
        check=True,
        capture_output=True,
    )
    return output


def compile_macos_capture_helper(output: Path) -> Path:
    """Compile the persistent public current-screen capture transport."""

    source = Path(__file__).parents[2] / "tools" / "macos_screen_capture.swift"
    if not source.is_file():
        raise FileNotFoundError(source)
    output.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "swiftc",
            "-O",
            "-parse-as-library",
            "-framework",
            "ScreenCaptureKit",
            "-framework",
            "CoreGraphics",
            str(source),
            "-o",
            str(output),
        ],
        check=True,
        capture_output=True,
    )
    return output


def compile_macos_av_capture_helper(output: Path) -> Path:
    """Compile the ScreenCaptureKit window tap that also emits public PCM."""

    source = Path(__file__).parents[2] / "tools" / "macos_window_av_capture.swift"
    if not source.is_file():
        raise FileNotFoundError(source)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "swiftc",
            "-O",
            "-parse-as-library",
            "-framework",
            "AVFoundation",
            "-framework",
            "CoreGraphics",
            "-framework",
            "CoreImage",
            "-framework",
            "CoreMedia",
            "-framework",
            "CoreVideo",
            "-framework",
            "ScreenCaptureKit",
            str(source),
            "-o",
            str(output),
        ],
        check=True,
        capture_output=True,
    )
    return output


def run_physical_brainworkshop_lifetime(
    machine: SourcePreservingTemporalMachine,
    config: PhysicalBrainWorkshopConfig,
    *,
    seconds: float,
    seed: int = 17,
    start_session: bool = False,
) -> PhysicalBrainWorkshopReport:
    """Run one bounded wall-clock lifetime against a frontmost real window."""

    if seconds <= 0.0:
        raise ValueError("physical lifetime must be positive")
    runtime, window, capture = build_physical_brainworkshop_runtime(
        machine, config, seed=seed
    )
    try:
        results = []
        if start_session:
            # Establish the ready-screen sensory baseline before the ordinary
            # human-visible SPACE action changes the interface.
            results.append(runtime.tick(time.monotonic()))
            window.press((" ",))
        started = time.monotonic()
        period = 1.0 / config.tick_hz
        next_tick = started
        while True:
            now = time.monotonic()
            if now - started >= seconds:
                break
            if now < next_tick:
                time.sleep(next_tick - now)
                now = time.monotonic()
            results.append(runtime.tick(now))
            next_tick += period
            next_tick = max(next_tick, now)
    finally:
        close_capture = getattr(capture, "close", None)
        if close_capture is not None:
            close_capture()
        close_output = getattr(runtime.input_device.output, "close", None)
        if close_output is not None:
            close_output()
    rewards: list[float] = []
    evidence: list[tuple[str, ...]] = []
    for result in results:
        for resolved in result.resolved_outcomes:
            if bool(resolved.event.present.item()):
                rewards.append(float(resolved.event.reward.item()))
                frame_evidence = getattr(resolved.event, "evidence", None)
                if frame_evidence is None:
                    raise RuntimeError(
                        "physical outcome escaped without public evidence"
                    )
                evidence.append(frame_evidence.frame_digests)
    frontend = runtime.input_device.frontend
    event_payloads = tuple(
        tuple(float(value) for value in payload)
        for payload in getattr(frontend, "emitted_payloads", ())
    )
    total_seconds = sorted(result.total_seconds for result in results)

    def latency_percentile(fraction: float) -> float | None:
        if not total_seconds:
            return None
        index = min(len(total_seconds) - 1, int(fraction * len(total_seconds)))
        return total_seconds[index]

    return PhysicalBrainWorkshopReport(
        ticks=len(results),
        input_events=sum(result.input_event_count for result in results),
        unique_public_outcomes=sum(result.outcome_bit_count for result in results),
        optimizer_updates=machine.optimizer_updates,
        program_file_updates=getattr(machine, "program_file_updates", 0),
        emitted_actions=sum(len(result.emitted_receipts) for result in results),
        deadline_misses=sum(result.deadline_missed for result in results),
        elapsed_seconds=time.monotonic() - started,
        tick_hz=config.tick_hz,
        action_delay_seconds=config.action_delay_seconds,
        capture_backend=config.capture_backend,
        rewards=tuple(rewards),
        actions=tuple(
            int(receipt.action.item())
            for result in results
            for receipt in result.emitted_receipts
        ),
        propensities=tuple(
            float(receipt.propensity.item())
            for result in results
            for receipt in result.emitted_receipts
        ),
        evidence_digests=tuple(evidence),
        event_payloads=event_payloads,
        evidence_archive=(
            None
            if config.evidence_directory is None
            else str(config.evidence_directory)
        ),
        total_seconds_p50=latency_percentile(0.50),
        total_seconds_p99=latency_percentile(0.99),
    )


def save_physical_report(report: PhysicalBrainWorkshopReport, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report.as_dict(), indent=2, sort_keys=True) + "\n")
