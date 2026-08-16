from __future__ import annotations

from pathlib import Path
from typing import Any

from experiments.brainworkshop_canonical.canonical_live_machine import (
    run_canonical_neural_workshop_live_lifetime,
)
from experiments.brainworkshop_canonical.cross_task_live_transfer import (
    run_live_cross_task_transfer,
)
from experiments.brainworkshop_canonical.live_operator_transfer import (
    run_live_operator_transfer,
)
from experiments.brainworkshop_canonical.maze_environment import sample_maze_task
from experiments.brainworkshop_canonical.maze_transfer import (
    EVENT_WIDTH,
    _run_cross_task_maze,
    build_event_dictionary,
)
from experiments.brainworkshop_canonical.neural_workshop_live import (
    NeuralWorkshopLiveConfig,
)
from experiments.brainworkshop_canonical.operator_world_transfer import verified_bundle
from experiments.brainworkshop_canonical.rendered_environment import (
    RenderedBrainWorkshopEncoders,
)
from experiments.brainworkshop_canonical.runner import CanonicalBrainWorkshopAgent


def _observation(
    sequence: int,
    *,
    outcome: dict[str, Any] | None = None,
    done: bool = False,
) -> dict[str, Any]:
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
        self._current = _observation(1)
        self._archive = {"stim-1": self._current["rgba"]}
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
                    "scalar": 1.0,
                    "evidence_digests": ["stim-1", "feedback-1"],
                    "receipt_id": 101,
                    "frame_seq": 2,
                    "timestamp_ns": 2_000_000,
                },
            )
        elif self._advances == 2:
            self._current = _observation(3)
        elif self._advances == 3:
            self._current = _observation(
                4,
                outcome={
                    "scalar": 1.0,
                    "evidence_digests": ["stim-3", "feedback-3"],
                    "receipt_id": 102,
                    "frame_seq": 4,
                    "timestamp_ns": 4_000_000,
                },
            )
        else:
            self._current = _observation(5, done=True)
        return self._current

    def close(self):
        self.closed = True


def test_canonical_live_workshop_then_maze_uses_one_core() -> None:
    environment = _Environment()

    def verifier(outcome, rgba, width, height, *, archive, receipt_ledger):
        del rgba, width, height, archive, receipt_ledger
        return outcome["receipt_id"] in {101, 102}

    agent = CanonicalBrainWorkshopAgent(
        symbol_count=4,
        event_width=EVENT_WIDTH,
        intention_width=4,
        feedback_width=8,
        n_back=1,
        reader_kind="context",
        seed=71,
    )
    report = run_canonical_neural_workshop_live_lifetime(
        agent,
        # The crops are deliberately non-overlapping and match the fake public
        # observation surface; no hidden task metadata enters the machine.
        config=NeuralWorkshopLiveConfig(
            active_cells=2,
            trials=2,
            event_width=EVENT_WIDTH,
            source_key_width=4,
            image_size=8,
            crop=(0.0, 0.25, 1.0, 1.0),
            instruction_crop=(0.0, 0.0, 1.0, 0.20),
            instruction_image_size=8,
            instruction_pool_size=4,
        ),
        seed=71,
        environment=environment,
        verifier=verifier,
        sample=False,
    )
    assert environment.closed
    assert report.emitted_actions == 2
    assert report.unique_verifier_bits == 2
    assert report.controller_frozen
    assert agent.intention_repertoire.record_count >= 2

    maze = sample_maze_task(seed=20_071, grid_size=7, minimum_distance=5)
    assert maze is not None
    encoders = RenderedBrainWorkshopEncoders.seeded(
        EVENT_WIDTH,
        source_key_width=4,
        seed=30_071,
    )
    dictionary = build_event_dictionary(maze, encoders)
    from experiments.brainworkshop_canonical.maze_transfer import SharedAmodalMazeAgent

    maze_agent = SharedAmodalMazeAgent(
        agent,
        encoders,
        dictionary,
        mode="workshop_warm",
        operator=verified_bundle(world_seed=10_071),
    )
    maze_result = _run_cross_task_maze(
        maze_agent,
        maze,
        seed=71,
        training_episodes=2,
        evaluation_episodes=1,
        steps=8,
        initial_verifier_bits=report.unique_verifier_bits,
    )
    assert maze_agent.core is agent
    assert maze_result["unique_verifier_bits"] > report.unique_verifier_bits
    assert report.controller_digest_before == maze_agent.controller_digest


def test_live_cross_task_runner_reuses_core_after_public_live_session(
    tmp_path: Path,
    monkeypatch,
) -> None:
    def verifier(outcome, rgba, width, height, *, archive, receipt_ledger):
        del rgba, width, height, archive, receipt_ledger
        return outcome["receipt_id"] in {101, 102}

    def build(_directory, _config, *, seed):
        del seed
        return _Environment(), verifier

    import experiments.brainworkshop_canonical.cross_task_live_transfer as module

    monkeypatch.setattr(module, "build_neural_workshop_environment", build)
    report = run_live_cross_task_transfer(
        Path(tmp_path),
        tmp_path / "report",
        seed=83,
        replicates=1,
        trials=2,
        maze_training_episodes=2,
        maze_evaluation_episodes=1,
        maze_steps=8,
    )
    assert report["claim_status"] == "development_diagnostic"
    assert report["shared_agent_boundary"][
        "one_controller_across_live_workshop_and_maze"
    ]
    row = report["replicates"][0]
    assert row["same_core_instance"]
    assert row["controller_unchanged"]
    assert row["intention_records_after_live_workshop"] >= 2
    assert row["live_workshop_after_maze"]["controller_frozen"]
    assert report["live_workshop_survives_maze_for_all_replicates"]
    assert (tmp_path / "report" / "live_cross_task_transfer.json").is_file()


def test_live_operator_transfer_stages_candidate_inside_rendered_loop(
    tmp_path: Path,
    monkeypatch,
) -> None:
    def verifier(outcome, rgba, width, height, *, archive, receipt_ledger):
        del rgba, width, height, archive, receipt_ledger
        return outcome["receipt_id"] in {101, 102}

    def build(_directory, _config, *, seed):
        del seed
        return _Environment(), verifier

    import experiments.brainworkshop_canonical.live_operator_transfer as module

    monkeypatch.setattr(module, "build_neural_workshop_environment", build)
    report = run_live_operator_transfer(
        Path(tmp_path),
        tmp_path / "report",
        seed=97,
        replicates=1,
        trials=2,
        source_maze_training_episodes=2,
        target_maze_training_episodes=2,
        maze_evaluation_episodes=1,
        maze_steps=8,
    )
    assert report["claim_status"] == "development_diagnostic"
    # The tiny fake maze does not earn a stable source prefix.  The important
    # live-loop property is fail-closed rebinding: no unverified operator is
    # sent to the target maze.
    assert not report["all_candidates_admitted"]
    assert report["controller_unchanged_for_all_replicates"]
    row = report["replicates"][0]
    assert row["same_core_instance"]
    assert row["operator_admission"]["observations"] == 1
    assert row["operator_admission"]["reason"] == "insufficient-stable-evidence"
    assert row["target_maze"]["operator_digest"] is None
    assert row["matched_control"]["target_maze"]["operator_digest"] is None
    assert row["matched_control"]["controller_unchanged"]
    assert row["live_workshop_after"]["controller_frozen"]
    assert (tmp_path / "report" / "live_operator_transfer.json").is_file()
