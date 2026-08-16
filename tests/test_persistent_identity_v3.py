from __future__ import annotations

import pytest
import torch

from neural_computer import PersistentCausalIdentityV3


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


def test_v3_rebinds_by_action_labelled_state_transitions() -> None:
    model = PersistentCausalIdentityV3()
    events, actions = _episode(controlled_track=0)
    first = model.resolve(events, actions, episode_id=0)
    events, actions = _episode(controlled_track=1)
    second = model.resolve(events, actions, episode_id=1)

    assert first.selected_slot.tolist() == [0]
    assert second.selected_slot.tolist() == [1]
    assert not bool(first.abstained[0])
    assert not bool(second.abstained[0])
    assert model.status == "active"
    assert model.support == 2


def test_v3_is_prefix_stable_within_an_episode() -> None:
    model = PersistentCausalIdentityV3()
    events, actions = _episode()
    first = model.resolve(events[:, :4], actions[:, :3], episode_id=10)
    second = model.resolve(events[:, :5], actions[:, :4], episode_id=10)

    assert not bool(first.abstained[0])
    assert torch.equal(first.selected_slot, second.selected_slot)
    assert not bool(second.abstained[0])
    assert model.support == 1


def test_v3_quarantines_on_known_transition_reversal() -> None:
    model = PersistentCausalIdentityV3()
    events, actions = _episode()
    model.resolve(events, actions, episode_id=0)
    reversed_events, actions = _episode(reverse=True)
    result = model.resolve(reversed_events, actions, episode_id=1)

    assert bool(result.abstained[0])
    assert model.status in {"active", "quarantined"}
    assert model.reason in {
        "state_evidence_insufficient",
        "low_applicability_or_margin",
        "causal_graph_contradiction",
    }


def test_v3_does_not_zero_fill_missing_evidence() -> None:
    model = PersistentCausalIdentityV3()
    events, actions = _episode()
    model.resolve(events, actions, episode_id=0)
    present = torch.ones(1, events.shape[1], events.shape[2], dtype=torch.bool)
    present[:, 3, 1] = False

    result = model.resolve(events, actions, event_present=present, episode_id=1)

    assert bool(result.abstained[0])
    assert model.status == "quarantined"
    assert model.reason == "missing_evidence"
    assert torch.equal(model.last_evidence, torch.zeros(1, 2))


def test_v3_abstains_for_exactly_equivalent_tracks() -> None:
    model = PersistentCausalIdentityV3()
    events, actions = _episode(duplicate=True)
    result = model.resolve(events, actions, episode_id=0)

    assert bool(result.abstained[0])
    assert model.status == "uninitialized"
    assert model.support == 0


def test_v3_updates_are_episode_idempotent() -> None:
    model = PersistentCausalIdentityV3()
    events, actions = _episode()
    first = model.resolve(events, actions, episode_id=11)
    second = model.resolve(events, actions, episode_id=11)

    assert not bool(first.abstained[0])
    assert torch.equal(first.selected_slot, second.selected_slot)
    assert model.support == 1


def test_v3_requires_single_external_stream() -> None:
    model = PersistentCausalIdentityV3()
    events, actions = _episode()
    with pytest.raises(ValueError, match="batch size one"):
        model.resolve(events.expand(2, -1, -1, -1), actions.expand(2, -1, -1))
