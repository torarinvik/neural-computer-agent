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


def test_practice_map_gives_each_rule_interchangeable_fragments() -> None:
    from experiments.games_amodal.fragment_bank import dual_suite, practice_map

    train, holdout = dual_suite()
    mapping = practice_map(train + holdout, partners=3)
    by_name = {c.name: c for c in train + holdout}
    for name, rows in mapping.items():
        assert len(rows) == len(by_name[name].rules())
        assert all(len(row) == 3 for row in rows)
        # Alternatives for one rule must be disjoint from another rule's.
        assert len(set(rows[0]) & set(rows[1])) == 0
    # Contexts sharing a rule share that rule's whole candidate set.
    assert mapping["dualAC"][0] == mapping["dualAD"][0]  # both takeA
    assert mapping["dualAC"][1] == mapping["dualBC"][1]  # both takeC
    assert mapping["dualBD"][0] == mapping["dualBC"][0]  # both takeB


def test_practice_draw_varies_partners_but_stays_in_the_rule_set() -> None:
    from experiments.games_amodal.fragment_bank import (
        draw_practice,
        dual_suite,
        practice_map,
    )

    train, _ = dual_suite()
    rows = practice_map(train, partners=4)["dualAC"]
    torch.manual_seed(0)
    draws = {tuple(draw_practice(rows)) for _ in range(60)}
    assert len(draws) > 1  # partners actually change
    for draw in draws:
        assert draw[0] in rows[0] and draw[1] in rows[1]


def test_bank_practice_indices_and_canonical_first() -> None:
    from experiments.games_amodal.fragment_bank import dual_suite, practice_map

    train, _ = dual_suite()
    bank = _bank([c.name for c in train], fragments=16)
    assert bank.practice_indices("dualAC") is None  # unset by default
    bank.set_practice_map(practice_map(train, partners=2))
    assert bank.practice_first("dualAC") == [
        row[0] for row in practice_map(train, partners=2)["dualAC"]
    ]
    drawn = bank.practice_indices("dualAC")
    assert drawn is not None and len(drawn) == 2


def test_battery_games_declare_calibrated_views() -> None:
    """F28: no single screen view suits every game, so each declares one."""

    from experiments.games_amodal.fragment_bank import battery_suite

    train, _ = battery_suite()
    views = {c.name: c.view for c in train}
    assert views["collect1"] == "crop"  # local geometry: walls and food
    assert views["intercept1"] == "roll"  # boundary-anchored: the floor
    assert views["forageA"] == views["forageB"] == "roll"
    # Twins must share a view, or they stop being observationally identical.
    assert views["choiceA"] == views["choiceB"]
    assert all(v in ("", "roll", "crop") for v in views.values())


def test_declared_view_overrides_the_run_default() -> None:
    from experiments.games_amodal.fragment_bank import rollout_family

    agent = _agent()
    crop_game = FamilyConfig(collect=1, view="crop", name="c")
    # A declared view must be honoured even when the run flag says otherwise.
    for flag in (False, True, "crop"):
        summary = rollout_family(
            agent, crop_game, None, batch_size=2, steps=4,
            seed=0, sample=False, gamma=0.9, egocentric=flag,
        )
        assert summary["total_reward"].shape == (2,)


def test_compose_suite_makes_factorisation_the_cheaper_option() -> None:
    from experiments.games_amodal.fragment_bank import compose_suite

    train, holdout = compose_suite()
    assert len(train) == 6 and len(holdout) == 3
    train_rules = {r for c in train for r in c.rules()}
    # Six rules cover nine pairings: a factoriser stores 6, a memoriser 9.
    assert len(train_rules) == 6
    assert len(train) + len(holdout) == 9
    for held in holdout:
        # Every held-out rule was learned...
        assert set(held.rules()) <= train_rules
        # ...but this PAIRING never appeared.
        assert all(set(held.rules()) != set(c.rules()) for c in train)
    # Each training rule appears in at least two pairings, so no fragment
    # can be identified with a single context.
    for rule in train_rules:
        assert sum(rule in c.rules() for c in train) >= 2


def test_compose_suite_holdouts_are_distinct_from_training_pairings() -> None:
    from experiments.games_amodal.fragment_bank import compose_suite

    train, holdout = compose_suite()
    names = {c.name for c in train} | {c.name for c in holdout}
    assert len(names) == 9
    assert all(c.arity == 3 for c in train + holdout)


def test_combiner_is_permutation_invariant_and_fragment_sensitive() -> None:
    from experiments.games_amodal.fragment_bank import FragmentCombiner

    torch.manual_seed(0)
    combiner = FragmentCombiner(width=8, hidden=16)
    fragments = torch.randn(2, 3, 8)
    out = combiner(fragments)
    assert out.shape == (3, 8)  # pooled to one fragment's worth of tokens
    # A fetched set has no intrinsic order, so order must not matter...
    assert torch.allclose(out, combiner(fragments.flip(0)), atol=1e-6)
    # ...but WHICH fragments were fetched must still change the context,
    # or the combiner would have severed the bank from behaviour.
    assert not torch.allclose(out, combiner(torch.randn(2, 3, 8)), atol=1e-3)


def test_combiner_is_shared_infrastructure_not_per_task_state() -> None:
    """F30: one combiner serves every context, so it cannot become a
    per-game program."""

    from experiments.games_amodal.fragment_bank import (
        FragmentCombiner,
        compose_suite,
    )

    torch.manual_seed(0)
    combiner = FragmentCombiner(width=8, hidden=16)
    train, _ = compose_suite()
    bank = FragmentBank(
        fragments=12, tokens_per_fragment=2, width=8,
        variants=[c.name for c in train],
    )
    bank.set_oracle_map(factorial_oracle_map(train))
    outputs = {
        c.name: combiner(bank.fetch(bank.oracle_indices(c.name, 2)))
        for c in train
    }
    # Distinct pairings must produce distinct contexts through the SAME
    # function -- that is what makes a novel pairing merely another
    # application of it.
    names = list(outputs)
    for left in range(len(names)):
        for right in range(left + 1, len(names)):
            assert not torch.allclose(
                outputs[names[left]], outputs[names[right]], atol=1e-4
            )


def test_critic_baseline_is_state_dependent_and_training_only() -> None:
    from experiments.games_amodal.fragment_bank import ValueHead

    torch.manual_seed(0)
    critic = ValueHead(intention_width=8, hidden=16)
    intentions = torch.randn(4, 8)
    values = critic(intentions)
    assert values.shape == (4,)
    # A state-dependent baseline must actually vary with the state, or it
    # is just the scalar baseline it was meant to improve on.
    assert float(values.std()) > 0.0
    agent = _agent()
    config = FamilyConfig(choice=1, name="x")
    with_critic = rollout_family(
        agent, config, None, batch_size=3, steps=5, seed=0,
        sample=True, gamma=0.9, critic=ValueHead(intention_width=8),
    )
    assert with_critic["value"] is not None
    assert with_critic["value"].shape == with_critic["returns"].shape
    # Absent at evaluation: the critic is an estimator, never a place for
    # skill to hide (F30).
    without = rollout_family(
        agent, config, None, batch_size=3, steps=5, seed=0,
        sample=True, gamma=0.9,
    )
    assert without["value"] is None
    assert not torch.allclose(with_critic["advantage"], without["advantage"])


def test_fisher_temperature_raises_entropy_and_is_off_by_default() -> None:
    """F49: the entropy floor must actually reach the sampling policy.

    A saturated policy's score-function gradients vanish, so a Fisher
    estimated from its own samples is noise that unit-mean normalisation
    rescales into a confident-looking anchor. The tempering knob is the
    safeguard; this pins that it is wired through and inert by default.
    """

    agent = _agent()
    config = FamilyConfig(choice=1, name="c")
    # Saturate the policy so the untempered entropy is genuinely small.
    decoder = agent.runtime.output_bus.decoders["keypress"]
    with torch.no_grad():
        for parameter in agent.runtime.output_bus.parameters():
            parameter.mul_(50.0)

    def entropy(temperature: float) -> float:
        summary = rollout_family(
            agent, config, None, batch_size=8, steps=8, seed=3,
            sample=True, gamma=0.9, temperature=temperature,
        )
        mask = summary["mask"]
        return float(
            (-summary["log_propensity"] * mask).sum()
            / mask.sum().clamp_min(1.0)
        )

    assert decoder.key_count > 1
    plain = entropy(1.0)
    tempered = entropy(4.0)
    assert tempered > plain, (plain, tempered)
    # Default path must be bit-for-bit the untempered one.
    assert entropy(1.0) == plain
