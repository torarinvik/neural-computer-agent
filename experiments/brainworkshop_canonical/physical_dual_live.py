"""Optional Dual I/O: Neural Workshop pixels plus ScreenCaptureKit PCM.

Position remains the visible play-field crop. Audio is the window's public
waveform, not a letter ID. Missing or silent Dual audio fails closed. Packed
actions use the two public keys: A for position and L for sound. Measured Dual
training uses the Neural Workshop gym, not this desktop tap.
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
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
    ExternalTemporalProgramBank,
    HumanParityLiveDevice,
    LiveActionReceipt,
    MacOSApplicationWindow,
    MacOSVirtualKeyOutput,
    NativeMacOSWindowAVCapture,
    NormalizedRegion,
    ProjectedScreenPulseFrontend,
    PublicWaveformEncoder,
    VisibleColorOutcomeReader,
    compose_recursive_temporal_program,
    one_hot_temporal_address_artifact,
    pcm_rms,
    recursive_temporal_primitive,
)

from .controller_pretraining import (
    build_recursive_temporal_program_machine,
    load_temporal_controller_artifact,
)
from .physical_live import (
    PhysicalBrainWorkshopReport,
    compile_macos_av_capture_helper,
    compile_macos_keypress_helper,
)
from .rendered_live import SourcePreservingTemporalMachine

PHYSICAL_DUAL_SCHEMA = "neural-computer.brainworkshop-physical-dual-live.v1"
POSITION_KEY_CODE = 0
SOUND_KEY_CODE = 37
SPACE_KEY_CODE = 49
MANUAL_KEY_CODE = 46
NBACK_DOWN_KEY_CODE = 122
NBACK_UP_KEY_CODE = 120
SILENCE_RMS = 0.008


@dataclass(frozen=True)
class PhysicalDualBrainWorkshopConfig:
    application: str = "Python"
    title_contains: str = "Neural Workshop"
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


def prepare_dual_ready_screen(
    window: MacOSApplicationWindow,
    keypress_helper: Path,
    *,
    n_back: int,
) -> None:
    """Leave the title screen, enter manual mode, and set public n-back."""

    if n_back < 1:
        raise ValueError("ready-screen n-back must be positive")
    output = MacOSVirtualKeyOutput(
        window,
        keypress_helper,
        {
            0: (SPACE_KEY_CODE,),
            1: (MANUAL_KEY_CODE,),
            2: (NBACK_DOWN_KEY_CODE,),
            3: (NBACK_UP_KEY_CODE,),
        },
    )
    try:
        dummy = torch.tensor([0], dtype=torch.int64)

        def press(action: int) -> None:
            receipt = LiveActionReceipt(
                receipt_id=action + 1,
                action=torch.tensor([action]),
                propensity=torch.tensor([1.0]),
                output_key="keyboard",
                emitted_at=time.monotonic(),
                model_version=0,
            )
            output.emit(dummy.new_tensor([action]), receipt)
            time.sleep(0.25)

        press(0)
        press(1)
        # Default Dual n-back is 2. One F1 reaches 1-back; F2 climbs.
        if n_back == 1:
            press(2)
        elif n_back > 2:
            for _ in range(n_back - 2):
                press(3)
    finally:
        output.close()


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
        process_aliases=("python", "Python"),
        query_helper=config.capture_helper,
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
    outcome_reader = dual_feedback_reader(config)
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


def dual_feedback_reader(
    config: PhysicalDualBrainWorkshopConfig,
) -> VisibleColorOutcomeReader:
    """Score Dual from the two public bottom labels.

    Neural Workshop paints each modality independently: green correct, red
    incorrect, blue missed-match. Any red or blue makes the packed trial
    wrong. Green without a reject color is a fully correct packed trial,
    including one explicit hit plus a correct rejection.
    """

    return VisibleColorOutcomeReader(
        region=config.feedback_region,
        negative_colors=((255, 64, 64), (64, 64, 255)),
        positive_colors=((0, 255, 0), (64, 255, 64)),
        neutral_is_positive=False,
        tolerance=config.feedback_tolerance,
        minimum_pixels=config.feedback_minimum_pixels,
        minimum_frames=config.feedback_minimum_frames,
        archive_directory=config.evidence_directory,
    )


def run_physical_dual_lifetime(
    machine: SourcePreservingTemporalMachine,
    config: PhysicalDualBrainWorkshopConfig,
    *,
    seconds: float,
    seed: int = 17,
    start_session: bool = False,
) -> PhysicalBrainWorkshopReport:
    """Run one bounded Dual lifetime against a frontmost Dual window."""

    if seconds <= 0.0:
        raise ValueError("physical Dual lifetime must be positive")
    runtime, window, capture = build_physical_dual_runtime(
        machine, config, seed=seed
    )
    try:
        results = []
        if start_session:
            results.append(runtime.tick(time.monotonic()))
            window.activate()
            # pyglet Dual ignores Space until the window has mouse focus.
            state = window.state()
            click_x = state.bounds[0] + state.bounds[2] // 2
            click_y = state.bounds[1] + state.bounds[3] // 2
            subprocess.run(
                [
                    "osascript",
                    "-e",
                    f'tell application "System Events" to click at {{{click_x}, {click_y}}}',
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            time.sleep(0.2)
            if config.keypress_helper is None:
                window.press((" ",))
            else:
                starter = MacOSVirtualKeyOutput(
                    window, config.keypress_helper, {0: (SPACE_KEY_CODE,)}
                )
                try:
                    receipt = LiveActionReceipt(
                        receipt_id=0,
                        action=torch.tensor([0]),
                        propensity=torch.tensor([1.0]),
                        output_key="keyboard",
                        emitted_at=time.monotonic(),
                        model_version=0,
                    )
                    starter.emit(torch.tensor([0], dtype=torch.int64), receipt)
                finally:
                    starter.close()
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
        capture.close()
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
                        "physical Dual outcome escaped without public evidence"
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
        capture_backend="sck-av",
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


def inherit_bank_program(
    machine: SourcePreservingTemporalMachine,
    bank_path: Path,
    *,
    slot: int = 0,
    learn: bool = False,
) -> dict[str, Any]:
    """Load an admitted temporal address file onto Dual I/O.

    Packed Dual actions are a runtime decoder adapter. They do not change
    the frozen controller digest, so ``AgentBrain.bank`` loads exactly.
    The curated bank is not mutated.
    """

    bank = ExternalTemporalProgramBank.load_bank(bank_path)
    if slot < 0 or slot >= bank.program_count:
        raise ValueError(f"bank slot {slot} is outside {bank.program_count} programs")
    artifact = bank.artifact(slot)
    if not machine.accepts_controller_digest(bank.controller_digest):
        raise ValueError("bank program targets another frozen controller")
    from .bank_program import install_temporal_artifact

    install_temporal_artifact(machine, bank, artifact)
    machine.learning_enabled = bool(learn)
    machine.sample = bool(learn)
    binding = (
        "exact"
        if bank.controller_digest == machine.controller_digest()
        else "historical_controller_alias"
    )
    return {
        "bank": str(bank_path),
        "slot": slot,
        "bank_digest": bank.digest(),
        "bank_controller_digest": bank.controller_digest,
        "machine_controller_digest": machine.controller_digest(),
        "inherited_program_digest": artifact.digest(),
        "program_length": artifact.program_length,
        "controller_binding": binding,
    }


def _load_composed_previous(
    machine: SourcePreservingTemporalMachine, *, depth: int
) -> None:
    if depth < 1:
        raise ValueError("Dual composition depth must be positive")
    primitive = recursive_temporal_primitive(
        one_hot_temporal_address_artifact(0, machine.max_history)
    )
    machine.load_recursive_program_artifact(
        compose_recursive_temporal_program(primitive, depth),
        controller_digest=machine.controller_digest(),
    )
    machine.learning_enabled = False
    machine.sample = False


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


def _build_dual_machine(
    controller_artifact: Path,
    *,
    learn: bool,
    learning_rate: float,
) -> SourcePreservingTemporalMachine:
    machine = build_recursive_temporal_program_machine(
        load_temporal_controller_artifact(controller_artifact),
        learning_rate=learning_rate,
        sample=learn,
        max_sources=2,
        pack_source_actions=True,
    )
    machine.learning_enabled = bool(learn)
    machine.sample = bool(learn)
    return machine


def _summarize_lifetime(
    report: PhysicalBrainWorkshopReport,
    machine: SourcePreservingTemporalMachine,
    *,
    label: str,
    n_back: int,
    depth: int,
    learn: bool,
) -> dict[str, Any]:
    return {
        "label": label,
        "n_back": n_back,
        "composition_depth": depth,
        "learn": learn,
        "ticks": report.ticks,
        "input_events": report.input_events,
        "unique_public_outcomes": report.unique_public_outcomes,
        "accuracy": (
            None
            if not report.rewards
            else sum(report.rewards) / len(report.rewards)
        ),
        "rewards": list(report.rewards),
        "actions": list(report.actions),
        "optimizer_updates": report.optimizer_updates,
        "program_file_updates": report.program_file_updates,
        "controller_frozen": True,
        "deadline_misses": report.deadline_misses,
        "elapsed_seconds": report.elapsed_seconds,
        "total_seconds_p50": report.total_seconds_p50,
        "total_seconds_p99": report.total_seconds_p99,
        "vision_events": len(report.event_payloads),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    repository = Path(__file__).parents[2]
    parser.add_argument(
        "--mode",
        choices=("probe", "execute", "learn"),
        default="probe",
    )
    parser.add_argument("--seconds", type=float, default=8.0)
    parser.add_argument("--sessions", type=int, default=1)
    parser.add_argument("--n-back", type=int, default=1)
    parser.add_argument("--compose-depth", type=int, default=None)
    parser.add_argument(
        "--bank",
        type=Path,
        default=(
            repository / "artifacts/checkpoints/AgentBrain.bank"
        ),
        help="continue this temporal bank instead of composing PREVIOUS",
    )
    parser.add_argument(
        "--bank-slot",
        type=int,
        default=1,
        help="temporal bank slot; 0 is Position 1-back, 1 is gym Dual 1-back",
    )
    parser.add_argument(
        "--search",
        action="store_true",
        help="select the Dual file from the bank on rendered Dual, then execute",
    )
    parser.add_argument(
        "--previous",
        action="store_true",
        help="ignore --bank and execute the composed PREVIOUS file",
    )
    parser.add_argument("--then-compose-2back", action="store_true")
    parser.add_argument("--start-session", action="store_true")
    parser.add_argument(
        "--prepare-nback",
        type=int,
        default=None,
        help="send Space, M, and F1/F2 so Dual starts at this public n-back",
    )
    parser.add_argument("--tick-hz", type=float, default=6.0)
    parser.add_argument("--program-learning-rate", type=float, default=0.3)
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
    keys = compile_macos_keypress_helper(arguments.keypress_helper)
    if arguments.prepare_nback is not None:
        prepare_dual_ready_screen(
            MacOSApplicationWindow(
                "python",
                title_contains="Neural Workshop",
                require_frontmost=True,
                process_aliases=("Python",),
                query_helper=helper,
            ),
            keys,
            n_back=arguments.prepare_nback,
        )
    if arguments.mode == "probe":
        window = MacOSApplicationWindow(
            "python",
            title_contains="Neural Workshop",
            require_frontmost=False,
            process_aliases=("Python",),
            query_helper=helper,
        )
        capture = NativeMacOSWindowAVCapture(window, helper, require_audio=True)
        ticks = max(2, math.ceil(arguments.seconds * arguments.tick_hz))
        report = run_physical_dual_loopback_probe(
            capture, ticks=ticks, tick_hz=arguments.tick_hz
        )
        if not report["audio_stream_active"]:
            text = json.dumps(report, indent=2, sort_keys=True) + "\n"
            print(text, end="")
            raise SystemExit(2)
    else:
        if arguments.n_back < 1 or arguments.sessions < 1:
            raise ValueError("Dual live sessions and n-back must be positive")
        config = PhysicalDualBrainWorkshopConfig(
            tick_hz=arguments.tick_hz,
            capture_helper=helper,
            keypress_helper=keys,
        )
        learn = arguments.mode == "learn"
        machine = _build_dual_machine(
            arguments.controller_artifact,
            learn=learn,
            learning_rate=arguments.program_learning_rate,
        )
        depth = (
            arguments.n_back
            if arguments.compose_depth is None
            else arguments.compose_depth
        )
        inheritance: dict[str, Any] | None = None
        if arguments.search and arguments.previous:
            raise ValueError("--search cannot be combined with --previous")
        if arguments.search and learn:
            raise ValueError("desktop Dual search executes a selected file; it is not a trainer")
        if arguments.previous:
            if not learn:
                _load_composed_previous(machine, depth=depth)
        elif arguments.search:
            from .execute_bank_slot import search_and_install

            bank = ExternalTemporalProgramBank.load_bank(arguments.bank)
            search = search_and_install(
                machine,
                bank,
                n_back=arguments.n_back,
                steps=24,
                seed=77,
            )
            winner = search["winner"]
            if winner["kind"] == "retrieve":
                inheritance = inherit_bank_program(
                    machine,
                    arguments.bank,
                    slot=int(winner["slots"][0]),
                    learn=False,
                )
            else:
                inheritance = {
                    "bank": str(arguments.bank),
                    "slot": list(winner["slots"]),
                    "program_length": int(
                        getattr(machine, "composition_depth", 1)
                    ),
                    "bank_controller_digest": bank.controller_digest,
                    "machine_controller_digest": machine.controller_digest(),
                    "controller_binding": "exact",
                }
            inheritance["search"] = search
            depth = int(inheritance["program_length"])
        else:
            inheritance = inherit_bank_program(
                machine,
                arguments.bank,
                slot=arguments.bank_slot,
                learn=learn,
            )
            depth = int(inheritance["program_length"])
        arms = []
        for session in range(arguments.sessions):
            lifetime = run_physical_dual_lifetime(
                machine,
                config,
                seconds=arguments.seconds,
                seed=17 + session,
                start_session=arguments.start_session and session == 0,
            )
            arms.append(
                _summarize_lifetime(
                    lifetime,
                    machine,
                    label=f"{arguments.mode}-n{arguments.n_back}-s{session + 1}",
                    n_back=arguments.n_back,
                    depth=1 if learn else depth,
                    learn=learn,
                )
            )
        if arguments.then_compose_2back:
            if arguments.previous:
                if not learn:
                    raise ValueError("--then-compose-2back with --previous needs learn")
                composed = compose_recursive_temporal_program(
                    machine.admitted_program_artifact(), 2
                )
            else:
                from .bank_program import compose_admitted_temporal

                composed = compose_admitted_temporal(
                    ExternalTemporalProgramBank.load_bank(arguments.bank),
                    (arguments.bank_slot, arguments.bank_slot),
                )
            machine.load_recursive_program_artifact(
                composed, controller_digest=machine.controller_digest()
            )
            composed_lifetime = run_physical_dual_lifetime(
                machine,
                config,
                seconds=arguments.seconds,
                seed=117,
                start_session=False,
            )
            arms.append(
                _summarize_lifetime(
                    composed_lifetime,
                    machine,
                    label="compose-2back",
                    n_back=2,
                    depth=2,
                    learn=False,
                )
            )
        report = {
            "schema": PHYSICAL_DUAL_SCHEMA,
            "mode": arguments.mode,
            "n_back": arguments.n_back,
            "controller_digest": machine.controller_digest(),
            "program_digest": machine.program_digest(),
            "optimizer_updates": machine.optimizer_updates,
            "program_file_updates": getattr(machine, "program_file_updates", 0),
            "inheritance": inheritance,
            "arms": arms,
        }
    text = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if arguments.report_out is None:
        print(text, end="")
    else:
        arguments.report_out.parent.mkdir(parents=True, exist_ok=True)
        arguments.report_out.write_text(text)


if __name__ == "__main__":
    main()
