import argparse

import pytest
import torch

from experiments.games_amodal.fragment_bank import (
    FragmentBank,
    dual_suite,
    factorial_oracle_map,
    has_positive_source,
    mastery,
    rollout_family,
    sample_selection,
    twins_suite,
    update_conflict,
)
from experiments.games_amodal.game_family import FamilyConfig
from experiments.games_amodal.shared_controller import SharedControllerAgent


def _bank(variants: list[str], fragments: int = 6) -> FragmentBank:
    torch.manual_seed(0)
    return FragmentBank(
        fragments=fragments,
        tokens_per_fragment=2,
        width=16,  # must match the agent's event width
        variants=variants,
    )


def _agent() -> SharedControllerAgent:
    torch.manual_seed(0)
    return SharedControllerAgent(
        event_width=16,
        intention_width=8,
        feedback_width=8,
        hidden=8,
        event_window_capacity=8,
        shared_drivers=True,
    )


def test_dual_suite_holds_out_a_novel_recombination() -> None:
    train, holdout = dual_suite()
    assert len(train) == 3 and len(holdout) == 1
    seen = {rule for config in train for rule in config.rules()}
    held = set(holdout[0].rules())
    # Every held-out rule was learned elsewhere, but never in this pairing.
    assert held <= seen
    assert all(set(config.rules()) != held for config in train)


def test_factorial_map_shares_one_fragment_per_shared_rule() -> None:
    train, holdout = dual_suite()
    mapping = factorial_oracle_map(train + holdout)
    by_name = {config.name: config for config in train + holdout}
    for left in mapping:
        for right in mapping:
            shared_rules = set(by_name[left].rules()) & set(
                by_name[right].rules()
            )
            shared_fragments = set(mapping[left]) & set(mapping[right])
            assert len(shared_fragments) == len(shared_rules)


def test_oracle_map_overrides_the_disjoint_default() -> None:
    bank = _bank(["a", "b"])
    assert bank.oracle_indices("a", 2) == [0, 1]
    bank.set_oracle_map({"a": [3, 4], "b": [3, 5]})
    assert bank.oracle_indices("a", 2) == [3, 4]
    assert bank.oracle_indices("b", 2) == [3, 5]
    with pytest.raises(ValueError, match="oracle map"):
        bank.oracle_indices("a", 3)


def test_twins_oracle_stays_disjoint_by_default() -> None:
    train, _ = twins_suite()
    bank = _bank([config.name for config in train])
    left = set(bank.oracle_indices(train[0].name, 2))
    right = set(bank.oracle_indices(train[1].name, 2))
    assert not (left & right)


def test_sample_selection_returns_distinct_fragments() -> None:
    logits = torch.randn(6)
    chosen, log_prob = sample_selection(logits, 3, greedy=False)
    assert len(set(chosen)) == 3
    assert torch.isfinite(log_prob)
    greedy, _ = sample_selection(logits, 3, greedy=True)
    assert greedy == torch.topk(logits, 3).indices.tolist()


def test_dual_variants_are_scored_by_rule_knowledge_not_reward() -> None:
    config = FamilyConfig(dual=1, name="dualAC")
    assert has_positive_source(config)
    summary = {
        "total_reward": torch.tensor([9.0, 9.0]),  # plenty of reward...
        "mask": torch.ones(2, 3),
        "rule_accuracy": torch.tensor([1.0, 0.5]),  # ...but one rule unknown
        "rule_engagement": torch.tensor([4.0, 4.0]),
    }
    assert mastery(summary, config) == pytest.approx(0.75)
    # An agent that refuses a trial kind gets no credit for it.
    refused = {
        **summary,
        "rule_accuracy": torch.tensor([1.0, 0.0]),
        "rule_engagement": torch.tensor([4.0, 0.0]),
    }
    assert mastery(refused, config) == pytest.approx(0.5)


def test_rollout_reports_per_rule_accuracy_only_for_dual() -> None:
    agent = _agent()
    dual = rollout_family(
        agent,
        FamilyConfig(dual=1, name="d"),
        None,
        batch_size=2,
        steps=6,
        seed=0,
        sample=False,
        gamma=0.9,
    )
    assert dual["rule_accuracy"] is not None
    assert dual["rule_accuracy"].shape == (2,)
    assert bool(((dual["rule_accuracy"] >= 0) & (dual["rule_accuracy"] <= 1)).all())
    choice = rollout_family(
        agent,
        FamilyConfig(choice=1, name="c"),
        None,
        batch_size=2,
        steps=6,
        seed=0,
        sample=False,
        gamma=0.9,
    )
    assert choice["rule_accuracy"] is None


def test_conflict_estimate_is_bounded_and_pair_keyed() -> None:
    train, _ = dual_suite()
    agent = _agent()
    bank = _bank([config.name for config in train])
    conflict = {
        frozenset((left.name, right.name)): 0.5
        for index, left in enumerate(train)
        for right in train[index + 1 :]
    }
    recent = {config.name: 0.8 for config in train}
    args = argparse.Namespace(
        fragments_per_variant=2,
        batch_size=2,
        steps=8,
        seed=0,
        gamma=0.9,
        conflict_decay=0.5,
    )
    before = dict(conflict)
    for update in range(6):
        update_conflict(
            agent, bank, train, conflict, recent, args=args, update=update
        )
    assert set(conflict) == set(before)
    assert all(0.0 <= value <= 1.0 for value in conflict.values())
    # A swap that costs the target its competence must register as conflict.
    assert any(conflict[key] != before[key] for key in conflict)


def test_conflict_estimate_stays_low_when_swapping_is_harmless() -> None:
    train, _ = dual_suite()
    agent = _agent()
    bank = _bank([config.name for config in train])
    conflict = {frozenset((train[0].name, train[1].name)): 0.5}
    # An untrained target has no competence to lose, so no swap can hurt it
    # and the estimator must decay toward "these two may share".
    recent = {config.name: 0.0 for config in train}
    args = argparse.Namespace(
        fragments_per_variant=2,
        batch_size=2,
        steps=8,
        seed=0,
        gamma=0.9,
        conflict_decay=0.5,
    )
    for update in range(4):
        update_conflict(
            agent, bank, train[:2], conflict, recent, args=args, update=update
        )
    assert conflict[frozenset((train[0].name, train[1].name))] < 0.2


def test_battery_suite_is_many_simple_contexts_with_a_recombination_holdout() -> None:
    from experiments.games_amodal.fragment_bank import battery_suite

    train, holdout = battery_suite()
    assert len(train) >= 6
    names = [config.name for config in train]
    assert len(names) == len(set(names))
    # Quantity comes from simplicity: every training game is one component
    # at level 1, so iteration budgets stay small.
    assert all(len(config.active()) == 1 for config in train)
    assert holdout[0].name == "dualBD"
    assert set(holdout[0].rules()) <= {
        rule for config in train for rule in config.rules()
    }


def test_battery_oracle_assignments_stay_disjoint_with_enough_fragments() -> None:
    from experiments.games_amodal.fragment_bank import battery_suite

    train, holdout = battery_suite()
    names = [config.name for config in train + holdout]
    bank = FragmentBank(
        fragments=2 * len(names),
        tokens_per_fragment=2,
        width=16,
        variants=names,
    )
    seen: set[int] = set()
    for name in names:
        indices = set(bank.oracle_indices(name, 2))
        assert not (indices & seen)
        seen |= indices
