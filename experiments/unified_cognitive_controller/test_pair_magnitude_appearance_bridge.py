from __future__ import annotations

import json

import pytest
import torch

from . import train_pair_magnitude_appearance_bridge as magnitude_bridge
from .environment import generate_lifetimes
from .train_pair_magnitude_appearance_bridge import (
    _annealed_gate_leak, _replay_specs, _shuffle_verifier_outcomes)


def test_magnitude_bridge_replays_every_inherited_capability() -> None:
    specs = _replay_specs((0.0, 0.15625))
    assert specs == (
        ("visible_pair_magnitude", "bars", 0.0),
        ("visible_pair_magnitude", "bars", 0.15625),
        ("pair_relation", "bars", None),
        ("pair_relation", "diamonds", None),
        ("pair_relation", "dot_pairs", None),
        ("binary_mapping", "bars", None),
        ("visible_context", "bars", None),
        ("visible_context_xor", "bars", None),
    )
    assert len(set(specs)) == len(specs)


def test_magnitude_bridge_rejects_duplicate_replay_rungs() -> None:
    with pytest.raises(ValueError, match="unique"):
        _replay_specs((0.0, 0.0))


def test_verifier_shuffle_preserves_pixels_and_outcome_marginal() -> None:
    batch = generate_lifetimes(
        64, 6, seed=21650, task="visible_pair_magnitude",
        appearance="bars", appearance_blend=0.203125)
    shuffled = _shuffle_verifier_outcomes(batch, seed=21651)
    assert torch.equal(shuffled.frames, batch.frames)
    assert torch.equal(
        shuffled.correct_actions.sort(dim=0).values,
        batch.correct_actions.sort(dim=0).values)
    assert not torch.equal(shuffled.correct_actions, batch.correct_actions)


def test_verifier_shuffle_falls_back_to_cells_for_duplicate_rows() -> None:
    from dataclasses import replace

    batch = generate_lifetimes(
        2, 6, seed=21652, task="visible_pair_magnitude",
        appearance="bars", appearance_blend=0.203125)
    batch = replace(
        batch, correct_actions=batch.correct_actions[:1].expand(2, -1))
    shuffled = _shuffle_verifier_outcomes(batch, seed=21653)
    assert torch.equal(shuffled.frames, batch.frames)
    assert torch.equal(
        shuffled.correct_actions.flatten().sort().values,
        batch.correct_actions.flatten().sort().values)
    assert not torch.equal(
        shuffled.correct_actions, batch.correct_actions)


def test_verifier_shuffle_rejects_a_constant_batch() -> None:
    from dataclasses import replace

    batch = generate_lifetimes(
        2, 6, seed=21654, task="visible_pair_magnitude",
        appearance="bars", appearance_blend=0.203125)
    batch = replace(
        batch, correct_actions=torch.zeros_like(batch.correct_actions))
    with pytest.raises(ValueError, match="cannot change"):
        _shuffle_verifier_outcomes(batch, seed=21655)


def test_fixed_gate_leak_schedule_is_prefix_invariant() -> None:
    short = [
        _annealed_gate_leak(0.05, optimizer_update=update, anneal_updates=16)
        for update in range(1, 9)
    ]
    long = [
        _annealed_gate_leak(0.05, optimizer_update=update, anneal_updates=16)
        for update in range(1, 25)
    ]
    assert short == long[:8]
    assert long[0] == pytest.approx(0.05)
    assert long[15:] == [0.0] * 9


def test_epochs_per_batch_reuse_experience_without_double_counting(
        monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    parser_values = [
        "train_pair_magnitude_appearance_bridge",
        "--parent", str(tmp_path / "parent.pt"),
        "--report", str(tmp_path / "report.json"),
        "--checkpoint-out", str(tmp_path / "child.pt"),
        "--candidate-checkpoint-out", str(tmp_path / "candidate.pt"),
        "--steps", "2",
        "--epochs-per-batch", "3",
        "--plasticity-mode", "refine",
        "--batch-size", "2",
        "--replay-batch-size", "2",
        "--test-lifetimes", "2",
        "--device", "cpu",
    ]
    monkeypatch.setattr("sys.argv", parser_values)

    class TinyController(torch.nn.Module):
        def __init__(self, **_):
            super().__init__()
            self.skill_adapter_gate_leak = 0.0
            self.skill_adapter_ablate_prior_read = False
            self.skill_adapter = torch.nn.Parameter(torch.zeros(()))

        def state_dict(self, *args, **kwargs):
            return {"skill_adapter": self.skill_adapter.detach().clone()}

        def load_state_dict(self, *_args, **_kwargs):
            return [], []

    # This accounting regression is intentionally isolated from the real
    # controller; all behavioral functions return differentiable constants.
    monkeypatch.setattr(
        magnitude_bridge, "UnifiedCognitiveController", TinyController)
    teacher = TinyController()
    payload = {
        "model_configuration": {"skill_adapter_widths": (1, 1)},
        "state_dict": teacher.state_dict(),
    }
    monkeypatch.setattr(
        magnitude_bridge, "_load", lambda *_: (payload, teacher))
    monkeypatch.setattr(
        magnitude_bridge, "_slot_prefixes",
        lambda _slot: ("skill_adapter",))
    generated = []

    def fake_generate(*args, **kwargs):
        generated.append((args, kwargs))
        return object()

    monkeypatch.setattr(magnitude_bridge, "generate_lifetimes", fake_generate)
    monkeypatch.setattr(
        magnitude_bridge, "_sensory_summary",
        lambda *_args, **_kwargs: [0.0, 1.0])
    monkeypatch.setattr(
        magnitude_bridge, "_pair_loss",
        lambda model, *_args, **_kwargs:
            (model.skill_adapter.square() + 1.0, 0.5, {
                "lifetime_accuracy_std": 0.0}))
    monkeypatch.setattr(
        magnitude_bridge, "_replay_loss_and_leakage",
        lambda model, *_args, **_kwargs:
            (model.skill_adapter.square(), model.skill_adapter.abs(), 0.5))
    accepted = {
        "gate": {"accepted": True},
    }
    monkeypatch.setattr(
        magnitude_bridge, "_target_evaluation",
        lambda *_args, **_kwargs: accepted)
    monkeypatch.setattr(
        magnitude_bridge, "evaluate",
        lambda *_args, **_kwargs: accepted)
    monkeypatch.setattr(
        magnitude_bridge, "_operation_cue_ablation_accuracy",
        lambda *_args, **_kwargs: 0.0)
    monkeypatch.setattr(
        magnitude_bridge, "_headline_accuracy",
        lambda _report: 1.0)

    magnitude_bridge.main()
    report = json.loads((tmp_path / "report.json").read_text())
    assert report["accounting"] == {
        "new_unique_lifetimes": 4,
        "new_verifier_bits": 24,
        "optimizer_lifetime_exposures": 96,
        "optimizer_updates": 6,
        "replay_specs": [
            ["visible_pair_magnitude", "bars", 0.0],
            ["pair_relation", "bars", None],
            ["pair_relation", "diamonds", None],
            ["pair_relation", "dot_pairs", None],
            ["binary_mapping", "bars", None],
            ["visible_context", "bars", None],
            ["visible_context_xor", "bars", None],
        ],
        "replay_streams": 7,
        "replay_unique_lifetimes": 28,
        "total_unique_lifetimes": 32,
        "total_unique_verifier_bits": 192,
    }
    # Two unique acquisition batches plus seven unique replay batches per
    # outer step. The three epochs consume no additional generated events.
    assert len(generated) == 16
    assert report["candidate_checkpoint_saved"]
    candidate = torch.load(
        tmp_path / "candidate.pt", map_location="cpu", weights_only=False)
    assert candidate["admission_status"] == (
        "unpromoted_pair_magnitude_prefix")
