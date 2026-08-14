from __future__ import annotations

from typing import Any

import pytest
import torch

from experiments.brainworkshop_canonical.neural_workshop_curriculum_pilot import (
    CurriculumPolicy,
    CurriculumSession,
    rung_mastered,
)
from experiments.brainworkshop_canonical.neural_workshop_live import (
    NeuralWorkshopInstructionEncoder,
    NeuralWorkshopIntervention,
    NeuralWorkshopLiveConfig,
    NeuralWorkshopRGBAEncoder,
    encode_instruction_context,
    run_neural_workshop_live_lifetime,
)
from experiments.brainworkshop_canonical.rendered_live import (
    SourcePreservingTemporalMachine,
)


def _observation(sequence: int, *, outcome: dict[str, Any] | None = None, done=False):
    width = height = 16
    pixels = bytearray([255] * (width * height * 4))
    offset = ((sequence * 3) % (width * height)) * 4
    pixels[offset : offset + 3] = b"\x00\x00\x00"
    result: dict[str, Any] = {
        "frame_seq": sequence,
        "timestamp_ns": sequence * 1_000_000,
        "width": width,
        "height": height,
        "rgba": bytes(pixels),
        "done": done,
    }
    if outcome is not None:
        result["outcome"] = outcome
    return result


class _Accounting:
    def snapshot(self):
        return {"logical_trials": 2}


class _Environment:
    n_actions = 1

    def __init__(self) -> None:
        first = _observation(1)
        self._current = first
        self._archive = {"stim-1": first["rgba"]}
        self._receipt_ledger: dict[int, object] = {}
        self._advances = 0
        self._receipt = 100
        self.accounting = _Accounting()
        self.closed = False

    def observe(self):
        return self._current

    def act(self, ports=None, logp=None):
        self._receipt += 1
        self._receipt_ledger[self._receipt] = {"ports": ports, "logp": logp}
        return {"ok": True, "receipt_id": self._receipt}

    def advance(self):
        self._advances += 1
        if self._advances == 1:
            self._current = _observation(
                2,
                outcome={
                    "scalar": -1.0,
                    "evidence_digests": ["stim-1", "feedback-1"],
                    "receipt_id": 101,
                    "frame_seq": 2,
                    "timestamp_ns": 2_000_000,
                },
            )
        elif self._advances == 2:
            self._current = _observation(3)
        elif self._advances == 3:
            # Correct-rejection silence is an explicit absent closure, not a
            # fabricated zero-reward verifier bit.
            self._current = _observation(4)
        else:
            self._current = _observation(5, done=True)
        return self._current

    def close(self):
        self.closed = True


def test_live_adapter_authenticates_signed_reward_and_drains_silence() -> None:
    environment = _Environment()
    calls = []

    def verifier(outcome, rgba, width, height, *, archive, receipt_ledger):
        calls.append((outcome, rgba, width, height, archive, receipt_ledger))
        return outcome["receipt_id"] == 101

    machine = SourcePreservingTemporalMachine(
        8,
        source_key_width=4,
        max_history=3,
        max_sources=1,
        action_count=2,
        sample=False,
    )
    report = run_neural_workshop_live_lifetime(
        machine,
        NeuralWorkshopLiveConfig(
            active_cells=2,
            trials=2,
            event_width=8,
            source_key_width=4,
            image_size=8,
            crop=(0.0, 0.25, 1.0, 1.0),
            instruction_crop=(0.0, 0.0, 1.0, 0.20),
            instruction_image_size=8,
            instruction_pool_size=4,
        ),
        seed=7,
        environment=environment,
        verifier=verifier,
        learn=True,
        sample=False,
    )

    assert environment.closed
    assert len(calls) == 1
    assert calls[0][4] is environment._archive
    assert calls[0][5] is environment._receipt_ledger
    assert report.emitted_actions == 2
    assert report.unique_verifier_bits == 1
    assert report.learner_outcome_bits == 1
    assert report.signed_scalars == (-1.0,)
    assert report.verifier_rewards == (0.0,)
    assert report.rewards == (0.0,)
    assert report.optimizer_updates == 1
    assert report.replayed_examples == 0
    assert report.ticks == 3


def test_live_interventions_preserve_verifier_truth_and_audit_execution() -> None:
    machine = SourcePreservingTemporalMachine(
        8,
        source_key_width=4,
        max_history=3,
        max_sources=1,
        action_count=2,
        sample=False,
    )
    environment = _Environment()

    def verifier(outcome, rgba, width, height, *, archive, receipt_ledger):
        del rgba, width, height, archive, receipt_ledger
        return outcome["receipt_id"] == 101

    report = run_neural_workshop_live_lifetime(
        machine,
        NeuralWorkshopLiveConfig(
            active_cells=2,
            trials=2,
            event_width=8,
            source_key_width=4,
            image_size=8,
            crop=(0.0, 0.25, 1.0, 1.0),
            instruction_crop=(0.0, 0.0, 1.0, 0.20),
            instruction_image_size=8,
            instruction_pool_size=4,
        ),
        seed=11,
        environment=environment,
        verifier=verifier,
        intervention=NeuralWorkshopIntervention(
            action="reversed",
            reward="missing",
            reset_history_each_tick=True,
            seed=11,
        ),
    )

    assert report.verifier_rewards == (0.0,)
    assert report.rewards == (0.0,)
    assert report.unique_verifier_bits == 1
    assert report.learner_outcome_bits == 0
    assert report.optimizer_updates == 0
    assert report.executed_actions == tuple(1 - value for value in report.actions)
    assert report.intervention == {
        "action": "reversed",
        "reward": "missing",
        "reset_history_each_tick": True,
        "seed": 11,
    }


def _summary(session: int, accuracy: float, bits: int = 8) -> CurriculumSession:
    return CurriculumSession(
        rung=0,
        active_cells=2,
        kind="train",
        seed=session,
        unique_verifier_bits=bits,
        positive_verifier_bits=round(bits * accuracy),
        accuracy=accuracy,
        optimizer_updates=0,
        program_file_updates=bits,
        replayed_examples=0,
        logical_trials=60,
        emitted_actions=60,
        wall_seconds=1.0,
        controller_frozen=True,
        report_path=f"session-{session}.json",
    )


def test_curriculum_requires_consecutive_supported_mastery() -> None:
    policy = CurriculumPolicy(stable_sessions=3)
    sessions = [_summary(1, 0.9), _summary(2, 0.7), _summary(3, 0.9)]
    assert not rung_mastered(sessions, policy)
    sessions.extend((_summary(4, 0.8), _summary(5, 0.85)))
    assert rung_mastered(sessions, policy)
    sessions[-1] = _summary(5, 1.0, bits=7)
    assert not rung_mastered(sessions, policy)


def _paint_band(
    width: int,
    height: int,
    *,
    top: float,
    bottom: float,
    color: tuple[int, int, int],
    mark: int | None = None,
) -> bytearray:
    pixels = bytearray([255] * (width * height * 4))
    y0 = int(top * height)
    y1 = max(y0 + 1, int(bottom * height))
    for row in range(y0, y1):
        for column in range(width):
            offset = (row * width + column) * 4
            pixels[offset : offset + 3] = bytes(color)
            pixels[offset + 3] = 255
    if mark is not None:
        mark_column = max(1, min(width - 2, mark))
        for row in range(y0, y1):
            offset = (row * width + mark_column) * 4
            pixels[offset : offset + 3] = b"\x00\x00\x00"
    return pixels


def test_instruction_encoder_stays_outside_the_play_field() -> None:
    config = NeuralWorkshopLiveConfig(event_width=8, source_key_width=4)
    play = NeuralWorkshopRGBAEncoder(config, seed=17)
    instruction = NeuralWorkshopInstructionEncoder(config)

    assert instruction.crop == config.instruction_crop
    assert instruction.crop[3] <= config.crop[1]
    assert not torch.equal(play.source_key.detach(), instruction.source_key.detach())
    with pytest.raises(ValueError, match="outside the play-field crop"):
        NeuralWorkshopLiveConfig(
            crop=(0.20, 0.05, 0.80, 0.80),
            instruction_crop=(0.10, 0.00, 0.90, 0.20),
        ).validate()


def test_instruction_and_play_field_respond_to_disjoint_pixels() -> None:
    config = NeuralWorkshopLiveConfig(
        event_width=8,
        source_key_width=4,
        image_size=16,
        crop=(0.20, 0.40, 0.80, 0.90),
        instruction_crop=(0.10, 0.00, 0.90, 0.20),
        instruction_image_size=16,
        instruction_pool_size=8,
    )
    play = NeuralWorkshopRGBAEncoder(config, seed=19)
    instruction = NeuralWorkshopInstructionEncoder(config)
    width = height = 32

    def observation(header_mark: int, field_mark: int) -> dict[str, Any]:
        pixels = _paint_band(
            width, height, top=0.40, bottom=0.90, color=(0, 0, 255), mark=field_mark
        )
        header = _paint_band(
            width, height, top=0.00, bottom=0.20, color=(255, 0, 0), mark=header_mark
        )
        for row in range(int(0.20 * height)):
            start = row * width * 4
            pixels[start : start + width * 4] = header[start : start + width * 4]
        return {
            "frame_seq": 1,
            "timestamp_ns": 1_000_000,
            "width": width,
            "height": height,
            "rgba": bytes(pixels),
            "done": False,
        }

    baseline = observation(6, 10)
    header_changed = observation(18, 10)
    field_changed = observation(6, 22)
    play_baseline = play.encode(baseline).payload.detach().clone()
    instruction_baseline = encode_instruction_context(baseline, instruction)
    play_header_changed = play.encode(header_changed).payload
    play_field_changed = play.encode(field_changed).payload
    instruction_header_changed = encode_instruction_context(
        header_changed, NeuralWorkshopInstructionEncoder(config)
    )
    instruction_field_changed = encode_instruction_context(
        field_changed, NeuralWorkshopInstructionEncoder(config)
    )

    assert torch.allclose(play_baseline, play_header_changed, atol=1e-5)
    assert float(torch.linalg.vector_norm(play_baseline - play_field_changed)) > 0.05
    assert float(
        torch.linalg.vector_norm(instruction_baseline - instruction_header_changed)
    ) > 0.05
    assert torch.allclose(instruction_baseline, instruction_field_changed, atol=1e-5)


def test_live_adapter_records_instruction_events_without_controller_input() -> None:
    environment = _Environment()
    machine = SourcePreservingTemporalMachine(
        8,
        source_key_width=4,
        max_history=3,
        max_sources=1,
        action_count=2,
        sample=False,
    )
    report = run_neural_workshop_live_lifetime(
        machine,
        NeuralWorkshopLiveConfig(
            active_cells=2,
            trials=2,
            event_width=8,
            source_key_width=4,
            image_size=8,
            crop=(0.0, 0.40, 1.0, 1.0),
            instruction_crop=(0.0, 0.0, 1.0, 0.20),
            instruction_image_size=8,
            instruction_pool_size=4,
        ),
        seed=7,
        environment=environment,
        verifier=lambda *args, **kwargs: True,
        learn=False,
        sample=False,
    )

    assert len(report.event_payloads) == 2
    assert len(report.instruction_payloads) == 2
    assert report.input_events == 4
    assert machine.max_sources == 1
