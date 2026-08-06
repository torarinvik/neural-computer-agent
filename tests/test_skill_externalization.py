import torch

from experiments.games_amodal.shared_controller import SharedControllerAgent
from experiments.games_amodal.skill_externalization import (
    artifact_events,
    ignorance_loss,
    rollout_with_artifact,
    train_externalized_skills,
)


def _agent() -> SharedControllerAgent:
    torch.manual_seed(0)
    return SharedControllerAgent(
        event_width=32,
        intention_width=16,
        feedback_width=8,
        hidden=16,
        event_window_capacity=8,
    )


def test_artifact_events_are_valid_batch_expanded_events() -> None:
    artifact = torch.randn(4, 32)
    events = artifact_events(artifact, batch_size=3)
    assert len(events) == 4
    for token, event in zip(artifact, events, strict=True):
        event.validate(width=32)
        assert event.payload.shape == (3, 32)
        assert torch.equal(event.payload[0], token)


def test_rollout_runs_with_and_without_artifact() -> None:
    agent = _agent()
    artifact = torch.randn(4, 32) * 0.1
    with_bank = rollout_with_artifact(
        agent, "snake", artifact, batch_size=2, steps=8, seed=1,
        sample=True, gamma=0.9,
    )
    without_bank = rollout_with_artifact(
        agent, "snake", None, batch_size=2, steps=8, seed=1,
        sample=True, gamma=0.9,
    )
    assert with_bank["total_reward"].shape == (2,)
    assert without_bank["logits"].shape[-1] == 4
    assert not torch.equal(with_bank["logits"], without_bank["logits"])


def test_ignorance_loss_is_zero_for_uniform_logits() -> None:
    logits = torch.zeros(2, 5, 4)
    mask = torch.ones(2, 5)
    assert float(ignorance_loss(logits, mask)) < 1e-8
    peaked = torch.zeros(2, 5, 4)
    peaked[..., 0] = 10.0
    assert float(ignorance_loss(peaked, mask)) > 0.5


def test_training_moves_both_artifacts_and_replays_nothing() -> None:
    agent = _agent()
    artifacts = {
        game: (torch.randn(4, 32) * 0.1).requires_grad_(True)
        for game in ("snake", "pong")
    }
    before = {game: a.detach().clone() for game, a in artifacts.items()}
    history = train_externalized_skills(
        agent,
        artifacts,
        updates=2,
        batch_size=4,
        steps=8,
        ignorance_steps=4,
        seed=2,
        gamma=0.9,
        learning_rate=1e-3,
        ignorance_weight=1.0,
        shuffle_rewards=False,
    )
    for game, artifact in artifacts.items():
        assert not torch.equal(before[game], artifact.detach())
    assert all(entry["replayed_examples"] == 0.0 for entry in history)
    assert all("decoy_loss" in entry for entry in history)
