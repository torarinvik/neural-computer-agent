from __future__ import annotations

from pathlib import Path

import pytest
import torch

from experiments.brainworkshop_canonical.physical_dual_live import (
    SILENCE_RMS,
    DualPulseFrontend,
    PhysicalDualBrainWorkshopConfig,
    dual_feedback_reader,
    dual_key_chords,
)
from experiments.brainworkshop_canonical.physical_live import (
    compile_macos_av_capture_helper,
)
from neural_computer import (
    CapturedScreenFrame,
    LiveActionReceipt,
    NativeMacOSWindowAVCapture,
    NormalizedRegion,
    ProjectedScreenPulseFrontend,
    PublicWaveformEncoder,
    discover_sck_window,
    encode_nca1_message,
    parse_nca1_message,
    pcm_rms,
)


def _tone(samples: int, *, period: int, amplitude: int = 12_000) -> bytes:
    values = []
    for index in range(samples):
        sign = 1 if (index // period) % 2 == 0 else -1
        values.append(int(sign * amplitude))
    return torch.tensor(values, dtype=torch.int16).numpy().tobytes()


def _frame(
    *,
    now: float,
    pulse: bool = False,
    pcm: bytes | None = None,
    stream_active: bool = True,
) -> CapturedScreenFrame:
    rgb = torch.full((3, 40, 60), 255, dtype=torch.uint8)
    if pulse:
        rgb[:, 5:20, 10:30] = 0
    audio = pcm if pcm is not None else _tone(800, period=8)
    return CapturedScreenFrame.from_public(
        rgb,
        captured_at=now,
        application="Public Test Window",
        title="Visible task",
        bounds=(10, 20, 60, 40),
        audio_pcm=audio,
        audio_rate=8_000,
        audio_channels=1,
        audio_sample_width=2,
        audio_stream_active=stream_active,
    )


def test_nca1_round_trip_preserves_pcm_and_flags() -> None:
    rgb = bytes([1, 2, 3, 4, 5, 6])
    pcm = _tone(16, period=4)
    payload = encode_nca1_message(
        width=2,
        height=1,
        rgb=rgb,
        pcm=pcm,
        audio_stream_active=True,
    )
    message = parse_nca1_message(payload)
    assert message["width"] == 2
    assert message["height"] == 1
    assert message["rgb"] == rgb
    assert message["pcm"] == pcm
    assert message["audio_stream_active"] is True


def test_inactive_audio_flag_is_preserved() -> None:
    payload = encode_nca1_message(
        width=1,
        height=1,
        rgb=bytes([9, 8, 7]),
        pcm=b"",
        audio_stream_active=False,
    )
    assert parse_nca1_message(payload)["audio_stream_active"] is False


def test_position_frames_remain_valid_without_audio() -> None:
    frame = CapturedScreenFrame.from_rgb(
        torch.full((3, 8, 8), 200, dtype=torch.uint8),
        captured_at=0.0,
        application="Public Test Window",
        title="Visible task",
        bounds=(0, 0, 8, 8),
    )
    assert frame.audio_pcm is None
    assert frame.validate() is frame


def test_dual_frontend_fails_closed_without_pcm() -> None:
    vision = ProjectedScreenPulseFrontend(
        8,
        region=NormalizedRegion(0.0, 0.0, 1.0, 0.75),
        change_threshold=0.01,
        refractory_seconds=0.2,
        seed=17,
    )
    audio = PublicWaveformEncoder(8, seed=17)
    frontend = DualPulseFrontend(vision, audio)
    blank = CapturedScreenFrame.from_rgb(
        torch.full((3, 40, 60), 255, dtype=torch.uint8),
        captured_at=0.0,
        application="Public Test Window",
        title="Visible task",
        bounds=(10, 20, 60, 40),
    )
    pulse_pixels = blank.rgb.clone()
    pulse_pixels[:, 5:20, 10:30] = 0
    pulse = CapturedScreenFrame.from_rgb(
        pulse_pixels,
        captured_at=0.1,
        application=blank.application,
        title=blank.title,
        bounds=blank.bounds,
    )
    frontend.observe(blank)
    with pytest.raises(RuntimeError, match="no ScreenCaptureKit audio tap"):
        frontend.observe(pulse)


def test_dual_frontend_fails_closed_on_silent_pcm() -> None:
    vision = ProjectedScreenPulseFrontend(
        8,
        region=NormalizedRegion(0.0, 0.0, 1.0, 0.75),
        change_threshold=0.01,
        refractory_seconds=0.2,
        seed=17,
    )
    audio = PublicWaveformEncoder(8, seed=17)
    frontend = DualPulseFrontend(vision, audio)
    silence = b"\x00\x00" * 800
    frontend.observe(_frame(now=0.0, pulse=False, pcm=silence))
    frontend.observe(_frame(now=0.1, pulse=True, pcm=silence))
    with pytest.raises(RuntimeError, match="no audible public PCM"):
        frontend.observe(_frame(now=0.4, pulse=True, pcm=silence))


def test_dual_frontend_emits_two_bound_events_after_one_silent_tick() -> None:
    vision = ProjectedScreenPulseFrontend(
        8,
        region=NormalizedRegion(0.0, 0.0, 1.0, 0.75),
        change_threshold=0.01,
        refractory_seconds=0.2,
        seed=17,
    )
    audio = PublicWaveformEncoder(8, seed=17)
    frontend = DualPulseFrontend(vision, audio)
    silence = b"\x00\x00" * 800
    frontend.observe(_frame(now=0.0, pulse=False, pcm=silence))
    held, held_boundary = frontend.observe(_frame(now=0.1, pulse=True, pcm=silence))
    events, boundary = frontend.observe(_frame(now=0.4, pulse=True, pcm=_tone(800, period=6)))

    assert held.payload.shape == (1, 0, 8)
    assert held_boundary is False
    assert boundary is True
    assert events.payload.shape == (1, 2, 8)
    assert events.source_key is not None
    assert not torch.equal(events.source_key[:, 0], events.source_key[:, 1])


def test_public_waveform_encoder_distinguishes_letter_envelopes() -> None:
    encoder = PublicWaveformEncoder(8, seed=17)
    first = encoder.encode(_frame(now=0.0, pcm=_tone(800, period=4)))
    second = encoder.encode(_frame(now=0.1, pcm=_tone(800, period=40)))
    distance = float(torch.linalg.vector_norm(first.payload - second.payload))
    assert distance > 0.05
    assert pcm_rms(_tone(800, period=4), sample_width=2) > SILENCE_RMS


def _dual_feedback_frame(
    now: float,
    left: tuple[int, int, int] | None,
    right: tuple[int, int, int] | None,
) -> CapturedScreenFrame:
    rgb = torch.full((3, 40, 60), 255, dtype=torch.uint8)
    if left is not None:
        rgb[:, 36:40, 4:18] = torch.tensor(left, dtype=torch.uint8)[:, None, None]
    if right is not None:
        rgb[:, 36:40, 42:56] = torch.tensor(right, dtype=torch.uint8)[:, None, None]
    return CapturedScreenFrame.from_rgb(
        rgb,
        captured_at=now,
        application="Public Test Window",
        title="Visible task",
        bounds=(10, 20, 60, 40),
    )


def _feedback_receipt() -> LiveActionReceipt:
    return LiveActionReceipt(
        receipt_id=1,
        action=torch.tensor([3]),
        propensity=torch.tensor([0.25]),
        output_key="keyboard",
        emitted_at=0.0,
        model_version=0,
    )


def test_dual_feedback_mixed_labels_are_exact_match_zero() -> None:
    reader = dual_feedback_reader(PhysicalDualBrainWorkshopConfig(capture_helper=Path(".")))
    green = (64, 255, 64)
    red = (255, 64, 64)
    oops = (64, 64, 255)
    reader.reset(_dual_feedback_frame(0.0, None, None))
    mixed = reader.close(_feedback_receipt(), _dual_feedback_frame(0.1, green, red))
    assert mixed.present.tolist() == [True]
    assert mixed.reward.tolist() == [0.0]

    reader.reset(_dual_feedback_frame(0.0, None, None))
    missed = reader.close(_feedback_receipt(), _dual_feedback_frame(0.1, green, oops))
    assert missed.present.tolist() == [True]
    assert missed.reward.tolist() == [0.0]


def test_dual_feedback_both_green_is_packed_correct() -> None:
    reader = dual_feedback_reader(PhysicalDualBrainWorkshopConfig(capture_helper=Path(".")))
    green = (64, 255, 64)
    reader.reset(_dual_feedback_frame(0.0, None, None))
    both = reader.close(_feedback_receipt(), _dual_feedback_frame(0.1, green, green))
    assert both.present.tolist() == [True]
    assert both.reward.tolist() == [1.0]


def test_packed_dual_keys_are_position_then_sound() -> None:
    chords = dual_key_chords()
    assert chords[0] == ()
    assert chords[1] == (0,)
    assert chords[2] == (37,)
    assert chords[3] == (0, 37)


def test_av_capture_parses_a_mock_helper(tmp_path: Path) -> None:
    rgb = bytes([10, 20, 30] * 4)
    pcm = _tone(8, period=2)
    message = encode_nca1_message(
        width=2, height=2, rgb=rgb, pcm=pcm, audio_stream_active=True
    )
    script = tmp_path / "fake_av_capture.py"
    script.write_text(
        "import sys\n"
        f"payload = {message!r}\n"
        "sys.stdin.read(1)\n"
        "sys.stdout.buffer.write(payload)\n"
        "sys.stdout.buffer.flush()\n"
    )
    executable = tmp_path / "fake_av_capture"
    executable.write_text(f"#!/usr/bin/env python3\n{script.read_text()}")
    executable.chmod(0o755)

    class _Window:
        def state(self):
            from neural_computer.human_io import MacOSWindowState

            return MacOSWindowState("Python", 1, "Neural Workshop", (0, 0, 2, 2), True)

        def lock_bounds(self, bounds):
            assert bounds == (0, 0, 2, 2)

    capture = NativeMacOSWindowAVCapture(_Window(), executable, require_audio=True)
    try:
        frame = capture.capture(1.5)
    finally:
        capture.close()
    assert frame.rgb.shape == (3, 2, 2)
    assert frame.audio_pcm == pcm
    assert frame.audio_stream_active is True


def test_sck_query_picks_the_brain_workshop_title(tmp_path: Path) -> None:
    payload = [
        {
            "pid": 11,
            "title": "Terminal",
            "application": "Terminal",
            "x": 0,
            "y": 0,
            "width": 800,
            "height": 600,
        },
        {
            "pid": 22,
            "title": "Neural Workshop 4.8.7",
            "application": "python",
            "x": 10,
            "y": 20,
            "width": 400,
            "height": 300,
        },
    ]
    helper = tmp_path / "fake_query"
    helper.write_text(
        "#!/usr/bin/env python3\n"
        "import json,sys\n"
        f"json.dump({payload!r}, sys.stdout)\n"
        "sys.stdout.write('\\n')\n"
    )
    helper.chmod(0o755)
    state = discover_sck_window(helper, title_contains="Neural Workshop")
    assert state.process_id == 22
    assert state.bounds == (10, 20, 400, 300)
    assert "Neural Workshop" in state.title


def test_screencapturekit_helper_compiles(tmp_path: Path) -> None:
    helper = compile_macos_av_capture_helper(tmp_path / "macos-window-av-capture")
    assert helper.is_file()
    assert helper.stat().st_size > 0
