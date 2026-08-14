"""Auditable live I/O for agents trained under human-visible rules.

This module does not know about games, rewards, or keyboard meanings.  It
provides a strict boundary in which every scalar learning outcome is backed by
the exact public screen frames from which it was derived.  Application-specific
regions, colors, and key chords remain replaceable frontend/backend policy.
"""

from __future__ import annotations

import hashlib
import json
import math
import struct
import subprocess
import tempfile
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, Self

import torch
from PIL import Image
from torch import nn

from .interface import AmodalEvent, AmodalEventCollection
from .live import LiveActionReceipt, LiveInputBatch, LiveOutcomeEvent

PUBLIC_OBSERVATION_EVIDENCE_SCHEMA = "neural-computer.public-observation-evidence.v1"
EVIDENCE_BOUND_OUTCOME_SCHEMA = "neural-computer.evidence-bound-outcome.v1"
CAPTURED_SCREEN_FRAME_SCHEMA = "neural-computer.captured-screen-frame.v1"


@dataclass(frozen=True)
class NormalizedRegion:
    """A public screen crop in top-left-origin normalized coordinates."""

    left: float
    top: float
    right: float
    bottom: float

    def validate(self) -> NormalizedRegion:
        values = (self.left, self.top, self.right, self.bottom)
        if not all(math.isfinite(value) for value in values):
            raise ValueError("normalized region must be finite")
        if not (
            0.0 <= self.left < self.right <= 1.0
            and 0.0 <= self.top < self.bottom <= 1.0
        ):
            raise ValueError("normalized region must be ordered within [0, 1]")
        return self

    def crop(self, rgb: torch.Tensor) -> torch.Tensor:
        self.validate()
        if rgb.ndim != 3 or rgb.shape[0] != 3:
            raise ValueError("screen pixels must have shape [3, height, width]")
        height, width = rgb.shape[1:]
        x0 = min(width - 1, int(self.left * width))
        x1 = max(x0 + 1, min(width, math.ceil(self.right * width)))
        y0 = min(height - 1, int(self.top * height))
        y1 = max(y0 + 1, min(height, math.ceil(self.bottom * height)))
        return rgb[:, y0:y1, x0:x1]


def _pixel_digest(rgb: torch.Tensor) -> str:
    pixels = rgb.detach().to(device="cpu", dtype=torch.uint8).contiguous()
    shape = b"x".join(str(value).encode() for value in pixels.shape)
    return hashlib.sha256(shape + b":" + pixels.numpy().tobytes()).hexdigest()


@dataclass(frozen=True)
class CapturedScreenFrame:
    """Exact public pixels plus audit-only capture provenance.

    Application identity and window geometry are never encoded into an amodal
    event.  They exist only to prevent capture or actuation against the wrong
    window and to make evidence reproducible.
    """

    rgb: torch.Tensor
    captured_at: float
    digest: str
    application: str
    title: str
    bounds: tuple[int, int, int, int]
    schema: str = CAPTURED_SCREEN_FRAME_SCHEMA

    @classmethod
    def from_rgb(
        cls,
        rgb: torch.Tensor,
        *,
        captured_at: float,
        application: str,
        title: str,
        bounds: tuple[int, int, int, int],
    ) -> CapturedScreenFrame:
        pixels = rgb.detach().to(device="cpu", dtype=torch.uint8).contiguous()
        return cls(
            rgb=pixels,
            captured_at=float(captured_at),
            digest=_pixel_digest(pixels),
            application=application,
            title=title,
            bounds=bounds,
        ).validate()

    def validate(self) -> CapturedScreenFrame:
        if self.schema != CAPTURED_SCREEN_FRAME_SCHEMA:
            raise ValueError(f"unsupported captured-frame schema: {self.schema}")
        if self.rgb.ndim != 3 or self.rgb.shape[0] != 3:
            raise ValueError("captured RGB frame must have shape [3, height, width]")
        if self.rgb.dtype != torch.uint8 or min(self.rgb.shape[1:]) < 1:
            raise ValueError("captured RGB frame must be non-empty uint8 pixels")
        if not math.isfinite(self.captured_at) or self.captured_at < 0.0:
            raise ValueError("capture time must be finite and non-negative")
        if not self.application:
            raise ValueError("captured frame needs an application identity")
        if len(self.bounds) != 4 or min(self.bounds[2:]) < 1:
            raise ValueError("captured frame needs positive window bounds")
        if self.digest != _pixel_digest(self.rgb):
            raise ValueError("captured frame digest does not match its pixels")
        return self


@dataclass(frozen=True)
class PublicObservationEvidence:
    """A closed observation window supporting one scalar outcome."""

    frame_digests: tuple[str, ...]
    started_at: float
    ended_at: float
    region: NormalizedRegion
    source: str = "public-screen"
    schema: str = PUBLIC_OBSERVATION_EVIDENCE_SCHEMA

    def validate(self) -> PublicObservationEvidence:
        if self.schema != PUBLIC_OBSERVATION_EVIDENCE_SCHEMA:
            raise ValueError(f"unsupported public-evidence schema: {self.schema}")
        if not self.frame_digests or any(
            len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
            for digest in self.frame_digests
        ):
            raise ValueError("public evidence needs SHA-256 screen-frame digests")
        if (
            not math.isfinite(self.started_at)
            or not math.isfinite(self.ended_at)
            or self.started_at < 0.0
            or self.ended_at < self.started_at
        ):
            raise ValueError("public evidence times are invalid")
        if not self.source:
            raise ValueError("public evidence source must be non-empty")
        self.region.validate()
        return self


@dataclass(frozen=True)
class EvidenceBoundOutcome(LiveOutcomeEvent):
    """A live scalar outcome whose entire evidence came from public input."""

    evidence: PublicObservationEvidence | None = None
    schema: str = EVIDENCE_BOUND_OUTCOME_SCHEMA

    def validate(self, *, batch_size: int) -> EvidenceBoundOutcome:
        if self.schema != EVIDENCE_BOUND_OUTCOME_SCHEMA:
            raise ValueError(f"unsupported evidence-bound schema: {self.schema}")
        base = LiveOutcomeEvent(
            receipt_id=self.receipt_id,
            reward=self.reward,
            present=self.present,
            observed_at=self.observed_at,
            confidence=self.confidence,
        )
        base.validate(batch_size=batch_size)
        if self.evidence is None:
            raise ValueError("human-parity outcomes require public evidence")
        self.evidence.validate()
        if self.observed_at != self.evidence.ended_at:
            raise ValueError("outcome time must close its public evidence window")
        return self


class ScreenFrameSource(Protocol):
    def capture(self, now: float) -> CapturedScreenFrame: ...


class ScreenFrontend(Protocol):
    event_width: int

    def observe(
        self, frame: CapturedScreenFrame
    ) -> tuple[AmodalEventCollection, bool]: ...


class PublicOutcomeReader(Protocol):
    def reset(self, frame: CapturedScreenFrame) -> None: ...

    def observe(self, frame: CapturedScreenFrame) -> None: ...

    def close(
        self, receipt: LiveActionReceipt, frame: CapturedScreenFrame
    ) -> EvidenceBoundOutcome: ...


class ReceiptOutput(Protocol):
    def emit(self, action: torch.Tensor, receipt: LiveActionReceipt) -> None: ...


class HumanParityLiveDevice:
    """Screen-in/action-out device with evidence-bound delayed outcomes.

    A frontend-provided public boundary (for example, a visible stimulus onset)
    closes the previous action.  Exactly one action may be pending.  This keeps
    causal matching explicit and prevents later UI state from being assigned to
    an arbitrary receipt.
    """

    batch_size = 1

    def __init__(
        self,
        frame_source: ScreenFrameSource,
        frontend: ScreenFrontend,
        outcome_reader: PublicOutcomeReader,
        output: ReceiptOutput,
    ) -> None:
        self.frame_source = frame_source
        self.frontend = frontend
        self.outcome_reader = outcome_reader
        self.output = output
        self.event_width = frontend.event_width
        self._pending: LiveActionReceipt | None = None
        self._have_window = False

    def poll(self, now: float) -> LiveInputBatch:
        frame = self.frame_source.capture(now).validate()
        events, boundary = self.frontend.observe(frame)
        outcomes: tuple[EvidenceBoundOutcome, ...] = ()
        if not self._have_window:
            self.outcome_reader.reset(frame)
            self._have_window = True
        else:
            self.outcome_reader.observe(frame)
        if boundary:
            if self._pending is not None:
                outcomes = (self.outcome_reader.close(self._pending, frame),)
                self._pending = None
            self.outcome_reader.reset(frame)
        return LiveInputBatch(events, outcomes, now)

    def emit(self, action: torch.Tensor, receipt: LiveActionReceipt) -> None:
        if self._pending is not None:
            raise RuntimeError("human-parity device permits one action per boundary")
        self.output.emit(action, receipt)
        self._pending = receipt


class ProjectedScreenPulseFrontend(nn.Module):
    """Generic directional pulse segmentation plus learned contrast projection.

    A dark-on-light or light-on-dark change gate decides only *when* a public
    frame becomes an event. Event content is the newly activated directional
    RGB contrast passed through a replaceable learned projection. Static window
    chrome and grid pixels therefore cannot dominate the learned event. No
    coordinates, labels, task IDs, or inferred symbols reach the controller.
    """

    def __init__(
        self,
        event_width: int,
        *,
        region: NormalizedRegion,
        source_key_width: int = 4,
        image_size: int = 36,
        change_threshold: float = 0.025,
        refractory_seconds: float = 0.5,
        dark_pulse: bool = True,
        seed: int = 0,
    ) -> None:
        super().__init__()
        if min(event_width, source_key_width, image_size) < 1:
            raise ValueError("screen frontend dimensions must be positive")
        if change_threshold <= 0.0 or refractory_seconds < 0.0:
            raise ValueError("screen pulse timing parameters are invalid")
        self.event_width = int(event_width)
        self.source_key_width = int(source_key_width)
        self.image_size = int(image_size)
        self.region = region.validate()
        self.change_threshold = float(change_threshold)
        self.refractory_seconds = float(refractory_seconds)
        self.dark_pulse = bool(dark_pulse)
        input_width = 3 * image_size * image_size
        generator = torch.Generator().manual_seed(seed)
        weight = torch.randn(event_width, input_width, generator=generator)
        weight = torch.nn.functional.normalize(weight, dim=1)
        self.projection = nn.Linear(input_width, event_width, bias=True)
        self.source_key = nn.Parameter(
            torch.randn(source_key_width, generator=generator)
        )
        with torch.no_grad():
            self.projection.weight.copy_(weight)
            self.projection.bias.zero_()
        self.normalization = nn.LayerNorm(event_width)
        self._previous_gate: torch.Tensor | None = None
        self._previous_resized: torch.Tensor | None = None
        self._last_boundary = -math.inf
        self.emitted_payloads: list[torch.Tensor] = []

    def _resize(self, pixels: torch.Tensor) -> torch.Tensor:
        value = pixels.to(dtype=torch.float32).unsqueeze(0) / 255.0
        return torch.nn.functional.interpolate(
            value,
            size=(self.image_size, self.image_size),
            mode="bilinear",
            align_corners=False,
        )

    def observe(self, frame: CapturedScreenFrame) -> tuple[AmodalEventCollection, bool]:
        pixels = self.region.crop(frame.rgb)
        resized = self._resize(pixels)
        gate = torch.nn.functional.interpolate(
            resized.mean(dim=1, keepdim=True),
            size=(18, 18),
            mode="bilinear",
            align_corners=False,
        ).detach()
        previous_gate = self._previous_gate
        previous_resized = self._previous_resized
        if previous_gate is None:
            signed_change = 0.0
        else:
            directional_change = (
                previous_gate - gate
                if self.dark_pulse
                else gate - previous_gate
            )
            # Measure newly activated pixels without letting simultaneous
            # release elsewhere cancel them. This still detects blank-to-pulse
            # transitions, and also detects a spatially moved pulse when a
            # short blank interval falls between screen samples.
            signed_change = float(directional_change.clamp_min(0.0).mean())
        self._previous_gate = gate
        self._previous_resized = resized.detach()
        boundary = (
            signed_change >= self.change_threshold
            and frame.captured_at - self._last_boundary >= self.refractory_seconds
        )
        if not boundary:
            return AmodalEventCollection.empty(1, self.event_width), False
        if previous_resized is None:
            raise RuntimeError("screen pulse boundary lacks a previous public frame")
        self._last_boundary = frame.captured_at
        directional_rgb = (
            previous_resized - resized
            if self.dark_pulse
            else resized - previous_resized
        ).clamp_min(0.0)
        payload = self.normalization(
            self.projection(directional_rgb.flatten(1))
        )
        self.emitted_payloads.append(payload[0].detach().cpu().clone())
        event = AmodalEvent(
            payload=payload,
            source_key=self.source_key.reshape(1, -1),
            timestamp=torch.tensor([frame.captured_at], dtype=payload.dtype),
            confidence=torch.ones(1, dtype=payload.dtype),
        )
        return AmodalEventCollection.from_events((event,), width=self.event_width), True


class VisibleColorOutcomeReader:
    """Interpret application-configured public feedback colors.

    Explicit positive feedback yields one and explicit negative feedback yields
    zero.  A neutral window is absent by default, so mere lack of an error is
    not converted into training signal.  Interfaces whose published human
    rules explicitly reward neutral windows may opt into that behavior.  Color
    meanings are external task instructions, not controller coordinates.
    """

    def __init__(
        self,
        *,
        region: NormalizedRegion,
        negative_colors: Sequence[tuple[int, int, int]],
        positive_colors: Sequence[tuple[int, int, int]] = (),
        neutral_is_positive: bool = False,
        tolerance: int = 24,
        minimum_pixels: int = 8,
        minimum_frames: int = 2,
        archive_directory: Path | None = None,
    ) -> None:
        if not negative_colors:
            raise ValueError("visible feedback needs at least one negative color")
        if tolerance < 0 or minimum_pixels < 1 or minimum_frames < 1:
            raise ValueError("visible feedback thresholds are invalid")
        self.region = region.validate()
        self.negative_colors = tuple(tuple(color) for color in negative_colors)
        self.positive_colors = tuple(tuple(color) for color in positive_colors)
        if any(
            len(color) != 3 or min(color) < 0 or max(color) > 255
            for color in self.negative_colors + self.positive_colors
        ):
            raise ValueError("feedback colors must be RGB triples")
        self.neutral_is_positive = bool(neutral_is_positive)
        self.tolerance = int(tolerance)
        self.minimum_pixels = int(minimum_pixels)
        self.minimum_frames = int(minimum_frames)
        self.archive_directory = archive_directory
        if archive_directory is not None:
            archive_directory.mkdir(parents=True, exist_ok=True)
        self._digests: list[str] = []
        self._started_at = 0.0
        self._negative_seen = False
        self._positive_seen = False
        self._last_observed_at: float | None = None

    def reset(self, frame: CapturedScreenFrame) -> None:
        self._digests = []
        self._started_at = frame.captured_at
        self._negative_seen = False
        self._positive_seen = False
        self._last_observed_at = None
        self.observe(frame)

    def observe(self, frame: CapturedScreenFrame) -> None:
        if (
            self._last_observed_at is not None
            and frame.captured_at < self._last_observed_at
        ):
            raise ValueError("public feedback frames must be chronological")
        self._digests.append(frame.digest)
        self._last_observed_at = frame.captured_at
        if self.archive_directory is not None:
            path = self.archive_directory / f"{frame.digest}.png"
            if not path.exists():
                array = frame.rgb.permute(1, 2, 0).contiguous().numpy()
                Image.fromarray(array).save(path, format="PNG")
        pixels = self.region.crop(frame.rgb).to(torch.int16).permute(1, 2, 0)
        for color in self.negative_colors:
            target = torch.tensor(color, dtype=torch.int16)
            matches = (pixels - target).abs().amax(dim=-1) <= self.tolerance
            if int(matches.sum()) >= self.minimum_pixels:
                self._negative_seen = True
        for color in self.positive_colors:
            target = torch.tensor(color, dtype=torch.int16)
            matches = (pixels - target).abs().amax(dim=-1) <= self.tolerance
            if int(matches.sum()) >= self.minimum_pixels:
                self._positive_seen = True

    def close(
        self, receipt: LiveActionReceipt, frame: CapturedScreenFrame
    ) -> EvidenceBoundOutcome:
        if self._last_observed_at != frame.captured_at:
            self.observe(frame)
        complete = len(self._digests) >= self.minimum_frames
        present = complete and (
            self._negative_seen
            or self._positive_seen
            or self.neutral_is_positive
        )
        reward = present and not self._negative_seen and (
            self._positive_seen or self.neutral_is_positive
        )
        evidence = PublicObservationEvidence(
            frame_digests=tuple(self._digests),
            started_at=self._started_at,
            ended_at=frame.captured_at,
            region=self.region,
        ).validate()
        return EvidenceBoundOutcome(
            receipt_id=receipt.receipt_id,
            reward=torch.tensor([float(reward)]),
            present=torch.tensor([present], dtype=torch.bool),
            observed_at=frame.captured_at,
            confidence=torch.tensor([1.0 if present else 0.0]),
            evidence=evidence,
        ).validate(batch_size=1)


@dataclass(frozen=True)
class MacOSWindowState:
    application: str
    process_id: int
    title: str
    bounds: tuple[int, int, int, int]
    frontmost: bool


class MacOSApplicationWindow:
    """Capture and actuate one allow-listed macOS application window."""

    def __init__(
        self,
        application: str,
        *,
        title_contains: str = "",
        require_frontmost: bool = True,
        state_refresh_seconds: float = 1.0,
    ) -> None:
        if not application:
            raise ValueError("macOS target application must be non-empty")
        if state_refresh_seconds < 0.0:
            raise ValueError("macOS state refresh interval cannot be negative")
        self.application = application
        self.title_contains = title_contains
        self.require_frontmost = bool(require_frontmost)
        self.state_refresh_seconds = float(state_refresh_seconds)
        self._cached_state: MacOSWindowState | None = None
        self._state_checked_at = -math.inf
        self._locked_bounds: tuple[int, int, int, int] | None = None

    def state(self) -> MacOSWindowState:
        script = (
            'var se=Application("System Events");'
            f"var p=se.applicationProcesses.byName({json.dumps(self.application)});"
            'if(!p.exists()) throw new Error("target application is not running");'
            'var ws=p.windows(); if(!ws.length) throw new Error("target has no window");'
            "var w=ws[0]; JSON.stringify({application:p.name(),title:w.name(),"
            "pid:p.unixId(),position:w.position(),size:w.size(),"
            "frontmost:p.frontmost()});"
        )
        completed: subprocess.CompletedProcess[str] | None = None
        for attempt in range(3):
            try:
                completed = subprocess.run(
                    ["osascript", "-l", "JavaScript", "-e", script],
                    check=True,
                    capture_output=True,
                    text=True,
                )
                break
            except subprocess.CalledProcessError:
                if attempt == 2:
                    raise
                time.sleep(0.02)
        if completed is None:
            raise RuntimeError("macOS target state query did not complete")
        value = json.loads(completed.stdout)
        position = tuple(int(item) for item in value["position"])
        size = tuple(int(item) for item in value["size"])
        state = MacOSWindowState(
            application=str(value["application"]),
            process_id=int(value["pid"]),
            title=str(value["title"]),
            bounds=(position[0], position[1], size[0], size[1]),
            frontmost=bool(value["frontmost"]),
        )
        if state.application != self.application:
            raise RuntimeError("macOS application identity changed")
        if self.title_contains and self.title_contains not in state.title:
            raise RuntimeError("macOS target window title does not match allow-list")
        if self.require_frontmost and not state.frontmost:
            raise RuntimeError(
                "refusing I/O because target application is not frontmost"
            )
        if min(state.bounds[2:]) < 1:
            raise RuntimeError("target application window has invalid bounds")
        if self._locked_bounds is not None and state.bounds != self._locked_bounds:
            raise RuntimeError("target application moved outside the locked capture")
        self._cached_state = state
        return state

    def require_fast_frontmost(self, process_id: int) -> None:
        """Verify the guarded process via native LaunchServices commands."""

        front = subprocess.run(
            ["lsappinfo", "front"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        information = subprocess.run(
            ["lsappinfo", "info", "-only", "pid", front],
            check=True,
            capture_output=True,
            text=True,
        ).stdout
        marker = '"pid"='
        lines = [line.strip() for line in information.splitlines() if marker in line]
        if len(lines) != 1:
            raise RuntimeError("could not identify the frontmost macOS process")
        observed = int(lines[0].split(marker, 1)[1])
        if observed != process_id:
            raise RuntimeError("refusing output because target is not frontmost")

    def lock_bounds(self, bounds: tuple[int, int, int, int]) -> None:
        if self._locked_bounds is not None and self._locked_bounds != bounds:
            raise RuntimeError("macOS target already has different locked bounds")
        self._locked_bounds = bounds

    def capture(self, now: float) -> CapturedScreenFrame:
        if (
            self._cached_state is None
            or now - self._state_checked_at >= self.state_refresh_seconds
        ):
            self._cached_state = self.state()
            self._state_checked_at = now
        state = self._cached_state
        with tempfile.TemporaryDirectory(prefix="neural-computer-capture-") as folder:
            output = Path(folder) / "frame.png"
            rectangle = ",".join(str(value) for value in state.bounds)
            subprocess.run(
                ["screencapture", "-x", "-R", rectangle, str(output)],
                check=True,
                capture_output=True,
            )
            image = Image.open(output).convert("RGB")
            rgb = torch.from_numpy(__import__("numpy").array(image, copy=True)).permute(
                2, 0, 1
            )
        return CapturedScreenFrame.from_rgb(
            rgb,
            captured_at=now,
            application=state.application,
            title=state.title,
            bounds=state.bounds,
        )

    def press(self, keys: Sequence[str]) -> None:
        self.state()
        for key in keys:
            if len(key) != 1 or not key.isprintable():
                raise ValueError(
                    "macOS text key output accepts one printable character"
                )
            script = f'tell application "System Events" to keystroke {json.dumps(key)}'
            subprocess.run(
                ["osascript", "-e", script],
                check=True,
                capture_output=True,
                text=True,
            )


class DiscreteKeyChordOutput:
    """External mapping from opaque action indices to public key chords."""

    def __init__(
        self,
        window: MacOSApplicationWindow,
        chords: Mapping[int, Sequence[str]],
    ) -> None:
        if not chords or min(chords) < 0:
            raise ValueError("key chord map needs non-negative action indices")
        self.window = window
        self.chords = {int(index): tuple(keys) for index, keys in chords.items()}

    def emit(self, action: torch.Tensor, receipt: LiveActionReceipt) -> None:
        del receipt
        if action.shape != (1,) or action.dtype not in (torch.int32, torch.int64):
            raise ValueError("physical key action must be one batch-one integer")
        index = int(action.item())
        if index not in self.chords:
            raise ValueError("opaque action is absent from the external chord map")
        self.window.press(self.chords[index])


class MacOSVirtualKeyOutput:
    """Low-latency external action-to-macOS-virtual-key backend."""

    def __init__(
        self,
        window: MacOSApplicationWindow,
        executable: Path,
        chords: Mapping[int, Sequence[int]],
    ) -> None:
        if not executable.is_file():
            raise ValueError("native macOS keypress helper does not exist")
        if not chords or min(chords) < 0:
            raise ValueError("virtual-key chord map needs non-negative actions")
        if any(
            key_code < 0 or key_code > 65_535
            for chord in chords.values()
            for key_code in chord
        ):
            raise ValueError("macOS virtual key codes must fit unsigned 16 bits")
        self.window = window
        self.executable = executable
        self.chords = {
            int(index): tuple(int(key_code) for key_code in chord)
            for index, chord in chords.items()
        }
        self._process = subprocess.Popen(
            [str(self.executable)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        )

    def emit(self, action: torch.Tensor, receipt: LiveActionReceipt) -> None:
        del receipt
        if action.shape != (1,) or action.dtype not in (torch.int32, torch.int64):
            raise ValueError("physical key action must be one batch-one integer")
        index = int(action.item())
        if index not in self.chords:
            raise ValueError("opaque action is absent from the virtual-key map")
        state = self.window._cached_state
        if state is None:
            state = self.window.state()
        self.window.require_fast_frontmost(state.process_id)
        if self._process.stdin is None or self._process.stdout is None:
            raise RuntimeError("native macOS keypress transport is unavailable")
        for key_code in self.chords[index]:
            self._process.stdin.write(f"{key_code}\n".encode())
            self._process.stdin.flush()
            if self._process.stdout.read(1) != b".":
                raise RuntimeError("native macOS keypress transport failed")

    def close(self) -> None:
        process = self._process
        if process.stdin is not None:
            process.stdin.close()
        try:
            process.wait(timeout=1.0)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=1.0)


class FFmpegMacOSWindowCapture:
    """Persistent low-latency public screen stream cropped to one window.

    AVFoundation captures the public display rather than application memory.
    Window bounds are locked when the stream starts. The native output backend
    revalidates the exact frontmost process before every later actuation.
    """

    def __init__(
        self,
        window: MacOSApplicationWindow,
        *,
        screen_input_index: int,
        frames_per_second: float,
        backing_scale: float = 2.0,
        ffmpeg: str = "ffmpeg",
    ) -> None:
        if screen_input_index < 0 or frames_per_second <= 0.0 or backing_scale <= 0.0:
            raise ValueError("FFmpeg screen capture settings are invalid")
        self.window = window
        self.screen_input_index = int(screen_input_index)
        self.frames_per_second = float(frames_per_second)
        self.backing_scale = float(backing_scale)
        self.ffmpeg = ffmpeg
        self._process: subprocess.Popen[bytes] | None = None
        self._state: MacOSWindowState | None = None
        self._pixel_size: tuple[int, int] | None = None

    def _start(self) -> None:
        state = self.window.state()
        self.window.lock_bounds(state.bounds)
        x, y, width, height = (
            round(value * self.backing_scale) for value in state.bounds
        )
        self._state = state
        self._pixel_size = (width, height)
        command = [
            self.ffmpeg,
            "-nostdin",
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "avfoundation",
            "-pixel_format",
            "bgr0",
            "-framerate",
            str(self.frames_per_second),
            "-i",
            f"{self.screen_input_index}:none",
            "-vf",
            f"crop={width}:{height}:{x}:{y}",
            "-r",
            str(self.frames_per_second),
            "-pix_fmt",
            "rgb24",
            "-f",
            "rawvideo",
            "pipe:1",
        ]
        self._process = subprocess.Popen(
            command,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        )

    def capture(self, now: float) -> CapturedScreenFrame:
        if self._process is None:
            self._start()
        if (
            self._process is None
            or self._process.stdout is None
            or self._state is None
            or self._pixel_size is None
        ):
            raise RuntimeError("FFmpeg screen stream did not start")
        width, height = self._pixel_size
        expected = width * height * 3
        data = self._process.stdout.read(expected)
        if len(data) != expected:
            return_code = self._process.poll()
            raise RuntimeError(
                f"FFmpeg screen stream ended early with status {return_code}"
            )
        numpy = __import__("numpy")
        rgb = torch.from_numpy(
            numpy.frombuffer(data, dtype=numpy.uint8).reshape(height, width, 3).copy()
        ).permute(2, 0, 1)
        return CapturedScreenFrame.from_rgb(
            rgb,
            captured_at=now,
            application=self._state.application,
            title=self._state.title,
            bounds=self._state.bounds,
        )

    def close(self) -> None:
        process = self._process
        self._process = None
        if process is None:
            return
        process.terminate()
        try:
            process.wait(timeout=2.0)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=2.0)

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *_error: object) -> None:
        self.close()


class NativeMacOSWindowCapture:
    """Persistent current-screen rectangle capture with no frame queue."""

    def __init__(
        self,
        window: MacOSApplicationWindow,
        executable: Path,
    ) -> None:
        if not executable.is_file():
            raise ValueError("native macOS capture helper does not exist")
        self.window = window
        self.executable = executable
        self._process: subprocess.Popen[bytes] | None = None
        self._state: MacOSWindowState | None = None

    @staticmethod
    def _read_exact(stream, count: int) -> bytes:
        chunks = bytearray()
        while len(chunks) < count:
            chunk = stream.read(count - len(chunks))
            if not chunk:
                raise RuntimeError("native macOS capture stream ended early")
            chunks.extend(chunk)
        return bytes(chunks)

    def _start(self) -> None:
        state = self.window.state()
        self.window.lock_bounds(state.bounds)
        self._state = state
        self._process = subprocess.Popen(
            [str(self.executable), *(str(value) for value in state.bounds)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        )

    def capture(self, now: float) -> CapturedScreenFrame:
        if self._process is None:
            self._start()
        process = self._process
        state = self._state
        if (
            process is None
            or process.stdin is None
            or process.stdout is None
            or state is None
        ):
            raise RuntimeError("native macOS capture transport is unavailable")
        process.stdin.write(b"\n")
        process.stdin.flush()
        width, height = struct.unpack("=II", self._read_exact(process.stdout, 8))
        raw = self._read_exact(process.stdout, width * height * 3)
        pixels = torch.frombuffer(bytearray(raw), dtype=torch.uint8)
        rgb = pixels.reshape(height, width, 3).permute(2, 0, 1).contiguous()
        return CapturedScreenFrame.from_rgb(
            rgb,
            captured_at=now,
            application=state.application,
            title=state.title,
            bounds=state.bounds,
        )

    def close(self) -> None:
        process = self._process
        self._process = None
        if process is None:
            return
        if process.stdin is not None:
            process.stdin.close()
        try:
            process.wait(timeout=1.0)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=1.0)

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *_error: object) -> None:
        self.close()
