from __future__ import annotations

import pytest
import torch

from neural_computer import PersistentCausalIdentityV2


def _episode(
    *,
    controlled_track: int = 0,
    reverse: bool = False,
    duplicate: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    actions = torch.tensor(
        [[[1.0, 0.0], [0.0, 1.0], [1.0, 0.0], [0.0, 1.0], [1.0, 0.0], [0.0, 1.0]]]
    )
    deltas = torch.zeros(1, 6, 2, 2)
    effect = 2.0 * actions[0, :, 0] + actions[0, :, 1]
    if reverse:
        effect = -effect
    deltas[0, :, controlled_track, 0] = effect
    if duplicate:
        deltas[0, :, 1 - controlled_track, 0] = effect
    events = torch.cat(
        [torch.zeros(1, 1, 2, 2), deltas.cumsum(dim=1)], dim=1
    )
    return events, actions


def test_persistent_identity_rebinds_the_model_to_a_new_slot() -> None:
    model = PersistentCausalIdentityV2()
    first_events, actions = _episode(controlled_track=0)
    first = model.resolve(first_events, actions, episode_id=0)
    second_events, actions = _episode(controlled_track=1)
    second = model.resolve(second_events, actions, episode_id=1)

    assert first.selected_slot.tolist() == [0]
    assert not bool(first.abstained[0])
    assert second.selected_slot.tolist() == [1]
    assert not bool(second.abstained[0])
    assert model.status == "active"
    assert model.support == 2


def test_persistent_identity_quarantines_on_reversal_and_relearns_from_fresh_data() -> None:
    model = PersistentCausalIdentityV2(recovery_episodes=2)
    events, actions = _episode()
    model.resolve(events, actions, episode_id=0)
    reversed_events, actions = _episode(reverse=True)
    contradicted = model.resolve(reversed_events, actions, episode_id=1)
    assert bool(contradicted.abstained[0])
    assert model.status == "quarantined"
    assert model.support == 1

    first_recovery = model.resolve(events, actions, episode_id=2)
    second_recovery = model.resolve(events, actions, episode_id=3)
    assert bool(first_recovery.abstained[0])
    assert bool(second_recovery.abstained[0])
    assert model.status == "active"
    assert model.reason == "relearned_requires_confirmation"
    confirmed = model.resolve(events, actions, episode_id=4)
    assert not bool(confirmed.abstained[0])


def test_persistent_identity_missing_evidence_is_not_zero_filled() -> None:
    model = PersistentCausalIdentityV2()
    events, actions = _episode()
    model.resolve(events, actions, episode_id=0)
    missing = torch.ones(1, events.shape[1], events.shape[2], dtype=torch.bool)
    missing[:, 3, 1] = False
    result = model.resolve(events, actions, event_present=missing, episode_id=1)

    assert bool(result.abstained[0])
    assert model.status == "quarantined"
    assert model.reason == "missing_evidence"
    assert model.support == 1
    assert torch.equal(model.last_evidence, torch.zeros(1, 2))


def test_persistent_identity_abstains_under_exact_dynamic_equivalence() -> None:
    model = PersistentCausalIdentityV2()
    events, actions = _episode(duplicate=True)
    result = model.resolve(events, actions, episode_id=0)

    assert bool(result.abstained[0])
    assert model.status == "uninitialized"
    assert model.support == 0


def test_persistent_identity_episode_id_makes_updates_idempotent() -> None:
    model = PersistentCausalIdentityV2()
    events, actions = _episode()
    first = model.resolve(events, actions, episode_id=11)
    second = model.resolve(events, actions, episode_id=11)

    assert not bool(first.abstained[0])
    assert not bool(second.abstained[0])
    assert model.support == 1


def test_persistent_identity_requires_single_external_stream() -> None:
    model = PersistentCausalIdentityV2()
    events, actions = _episode()
    with pytest.raises(ValueError, match="batch size one"):
        model.resolve(events.expand(2, -1, -1, -1), actions.expand(2, -1, -1))
