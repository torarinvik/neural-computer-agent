from __future__ import annotations

import torch

from neural_computer import (
    AmodalEvent,
    AmodalEventCollection,
    CapturedScreenFrame,
    EvidenceBoundOutcome,
    HumanParityLiveDevice,
    LiveActionReceipt,
    NormalizedRegion,
    ProjectedScreenPulseFrontend,
    VisibleColorOutcomeReader,
)


def _frame(
    value: int = 255,
    *,
    now: float = 0.0,
    color: tuple[int, int, int] | None = None,
) -> CapturedScreenFrame:
    rgb = torch.full((3, 40, 60), value, dtype=torch.uint8)
    if color is not None:
        rgb[:, 30:35, 10:20] = torch.tensor(color, dtype=torch.uint8)[:, None, None]
    return CapturedScreenFrame.from_rgb(
        rgb,
        captured_at=now,
        application="Public Test Window",
        title="Visible task",
        bounds=(10, 20, 60, 40),
    )


def _receipt(identifier: int = 1) -> LiveActionReceipt:
    return LiveActionReceipt(
        receipt_id=identifier,
        action=torch.tensor([0]),
        propensity=torch.tensor([0.25]),
        output_key="keyboard",
        emitted_at=0.0,
        model_version=0,
    )


def test_captured_frame_digest_authenticates_exact_public_pixels() -> None:
    frame = _frame()
    assert frame.validate() is frame
    altered = frame.rgb.clone()
    altered[0, 0, 0] = 0
    forged = CapturedScreenFrame(
        rgb=altered,
        captured_at=frame.captured_at,
        digest=frame.digest,
        application=frame.application,
        title=frame.title,
        bounds=frame.bounds,
    )

    try:
        forged.validate()
    except ValueError as error:
        assert "digest" in str(error)
    else:
        raise AssertionError("a forged public screen digest was accepted")


def test_visible_feedback_outcome_is_bound_to_all_public_screen_evidence(
    tmp_path,
) -> None:
    reader = VisibleColorOutcomeReader(
        region=NormalizedRegion(0.0, 0.5, 1.0, 1.0),
        negative_colors=((255, 64, 64), (64, 64, 255)),
        tolerance=0,
        minimum_pixels=8,
        minimum_frames=2,
        archive_directory=tmp_path,
    )
    first = _frame(now=0.0)
    negative = _frame(now=0.1, color=(255, 64, 64))
    final = _frame(now=0.2)
    reader.reset(first)
    reader.observe(negative)
    outcome = reader.close(_receipt(), final)

    assert isinstance(outcome, EvidenceBoundOutcome)
    assert outcome.present.tolist() == [True]
    assert outcome.reward.tolist() == [0.0]
    assert outcome.evidence is not None
    assert outcome.evidence.frame_digests == (
        first.digest,
        negative.digest,
        final.digest,
    )
    assert (tmp_path / f"{negative.digest}.png").is_file()


def test_visible_absence_of_error_is_reward_only_after_complete_window() -> None:
    reader = VisibleColorOutcomeReader(
        region=NormalizedRegion(0.0, 0.5, 1.0, 1.0),
        negative_colors=((255, 64, 64),),
        neutral_is_positive=True,
        minimum_frames=3,
    )
    first = _frame(now=0.0)
    second = _frame(now=0.1)
    reader.reset(first)
    incomplete = reader.close(_receipt(), second)
    assert incomplete.present.tolist() == [False]
    assert incomplete.reward.tolist() == [0.0]

    reader.reset(first)
    reader.observe(second)
    complete = reader.close(_receipt(), _frame(now=0.2))
    assert complete.present.tolist() == [True]
    assert complete.reward.tolist() == [1.0]


def test_projected_screen_frontend_emits_only_on_visible_pulse_onset() -> None:
    frontend = ProjectedScreenPulseFrontend(
        8,
        region=NormalizedRegion(0.0, 0.0, 1.0, 0.75),
        change_threshold=0.01,
        refractory_seconds=0.2,
        seed=17,
    )
    blank = _frame(now=0.0)
    pulse_pixels = blank.rgb.clone()
    pulse_pixels[:, 5:20, 10:30] = 0
    pulse = CapturedScreenFrame.from_rgb(
        pulse_pixels,
        captured_at=0.1,
        application=blank.application,
        title=blank.title,
        bounds=blank.bounds,
    )
    released = _frame(now=0.4)
    second_pulse = CapturedScreenFrame.from_rgb(
        pulse_pixels,
        captured_at=0.7,
        application=blank.application,
        title=blank.title,
        bounds=blank.bounds,
    )

    quiet, first_boundary = frontend.observe(blank)
    first, pulse_boundary = frontend.observe(pulse)
    held, held_boundary = frontend.observe(pulse)
    release, release_boundary = frontend.observe(released)
    second, second_boundary = frontend.observe(second_pulse)

    assert (
        quiet.payload.shape == held.payload.shape == release.payload.shape == (1, 0, 8)
    )
    assert first_boundary is False
    assert pulse_boundary is True
    assert held_boundary is False
    assert release_boundary is False
    assert second_boundary is True
    assert first.payload.shape == second.payload.shape == (1, 1, 8)
    assert torch.equal(first.payload, second.payload)
    assert first.source_key is not None


def test_projected_screen_frontend_detects_spatial_move_without_sampled_release() -> None:
    frontend = ProjectedScreenPulseFrontend(
        8,
        region=NormalizedRegion(0.0, 0.0, 1.0, 0.75),
        change_threshold=0.01,
        refractory_seconds=0.2,
        seed=17,
    )
    blank = _frame(now=0.0)
    left_pixels = blank.rgb.clone()
    left_pixels[:, 5:20, 5:20] = 0
    right_pixels = blank.rgb.clone()
    right_pixels[:, 5:20, 35:50] = 0
    left = CapturedScreenFrame.from_rgb(
        left_pixels,
        captured_at=0.1,
        application=blank.application,
        title=blank.title,
        bounds=blank.bounds,
    )
    right = CapturedScreenFrame.from_rgb(
        right_pixels,
        captured_at=0.4,
        application=blank.application,
        title=blank.title,
        bounds=blank.bounds,
    )

    frontend.observe(blank)
    _first, first_boundary = frontend.observe(left)
    moved, moved_boundary = frontend.observe(right)

    assert first_boundary is True
    assert moved_boundary is True
    assert moved.payload.shape == (1, 1, 8)
    assert not torch.equal(moved.payload, _first.payload)


class _Frames:
    def __init__(self, frames: list[CapturedScreenFrame]) -> None:
        self.frames = frames

    def capture(self, now: float) -> CapturedScreenFrame:
        frame = self.frames.pop(0)
        assert now == frame.captured_at
        return frame


class _Boundaries:
    event_width = 4

    def __init__(self, boundaries: list[bool]) -> None:
        self.boundaries = boundaries

    def observe(self, frame: CapturedScreenFrame) -> tuple[AmodalEventCollection, bool]:
        boundary = self.boundaries.pop(0)
        events = (
            AmodalEventCollection.from_events((AmodalEvent(torch.ones(1, 4)),), width=4)
            if boundary
            else AmodalEventCollection.empty(1, 4)
        )
        return events, boundary


class _Output:
    def __init__(self) -> None:
        self.receipts: list[LiveActionReceipt] = []

    def emit(self, action: torch.Tensor, receipt: LiveActionReceipt) -> None:
        assert action.tolist() == [0]
        self.receipts.append(receipt)


def test_human_parity_device_closes_neutral_receipt_without_reward() -> None:
    frames = [_frame(now=value) for value in (0.0, 0.1, 1.0)]
    reader = VisibleColorOutcomeReader(
        region=NormalizedRegion(0.0, 0.5, 1.0, 1.0),
        negative_colors=((255, 64, 64),),
        minimum_frames=2,
    )
    output = _Output()
    device = HumanParityLiveDevice(
        _Frames(frames), _Boundaries([True, False, True]), reader, output
    )

    first = device.poll(0.0)
    assert first.events.payload.shape == (1, 1, 4)
    device.emit(torch.tensor([0]), _receipt())
    assert device.poll(0.1).outcomes == ()
    final = device.poll(1.0)
    assert len(final.outcomes) == 1
    assert final.outcomes[0].receipt_id == 1
    assert final.outcomes[0].present.tolist() == [False]
    assert final.outcomes[0].reward.tolist() == [0.0]
    assert output.receipts[0].receipt_id == 1


def test_visible_positive_feedback_is_rewarded() -> None:
    reader = VisibleColorOutcomeReader(
        region=NormalizedRegion(0.0, 0.5, 1.0, 1.0),
        negative_colors=((255, 64, 64),),
        positive_colors=((64, 255, 64),),
        tolerance=0,
        minimum_pixels=8,
        minimum_frames=2,
    )
    first = _frame(now=0.0)
    reader.reset(first)
    positive = _frame(now=0.1, color=(64, 255, 64))

    outcome = reader.close(_receipt(), positive)

    assert outcome.present.tolist() == [True]
    assert outcome.reward.tolist() == [1.0]


def test_visible_feedback_accepts_multiple_public_positive_colors() -> None:
    reader = VisibleColorOutcomeReader(
        region=NormalizedRegion(0.0, 0.5, 1.0, 1.0),
        negative_colors=((255, 64, 64),),
        positive_colors=((0, 255, 0), (64, 255, 64)),
        tolerance=0,
        minimum_pixels=8,
        minimum_frames=2,
    )
    first = _frame(now=0.0)
    reader.reset(first)

    outcome = reader.close(
        _receipt(), _frame(now=0.1, color=(0, 255, 0))
    )

    assert outcome.present.tolist() == [True]
    assert outcome.reward.tolist() == [1.0]
