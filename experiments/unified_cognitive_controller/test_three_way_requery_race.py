import torch

from .train_safe_requery_adaptation import ActionValueHead
from .train_three_way_requery_race import (
    ComposedChampionThirdValueHead,
    FrozenChampionThirdValueHead,
    StoredCoreThirdValueHead,
    expand_advantage_head,
    expand_action_value_head,
    paired_multiaction_improvement,
)
from .train_shadow_compute_advantage import ComputeAdvantageHead


def test_expansion_preserves_known_action_values() -> None:
    torch.manual_seed(3)
    source = ActionValueHead(8)
    with torch.no_grad():
        source.network[-1].weight.normal_()
        source.network[-1].bias.normal_()
    features = torch.randn(11, 4)
    expanded = expand_action_value_head(source)
    source_values = source.q_values(features)
    expanded_values = expanded.q_values(features)
    assert torch.equal(source_values, expanded_values[:, :2])
    assert torch.allclose(
        expanded_values[:, 2], source_values.mean(1), atol=1e-6)


def test_multiaction_ips_detects_better_policy() -> None:
    attempted = torch.arange(3000).remainder(3)
    incumbent = torch.zeros_like(attempted)
    challenger = torch.ones_like(attempted)
    utility = (attempted == 1).float()
    evidence = paired_multiaction_improvement(
        incumbent, challenger, attempted, utility)
    assert evidence["estimated_improvement"] > 0.9
    assert evidence["lower_95"] > 0


def test_advantage_expansion_preserves_champion_decisions() -> None:
    torch.manual_seed(9)
    source = ComputeAdvantageHead(8)
    features = torch.randn(31, 4)
    expanded = expand_advantage_head(source)
    assert torch.equal(
        (source(features) > 0).long(),
        expanded(features))


def test_frozen_champion_preserves_old_policy_and_only_trains_new_value() -> None:
    torch.manual_seed(10)
    source = ComputeAdvantageHead(8)
    features = torch.randn(31, 4)
    expanded = FrozenChampionThirdValueHead(source)
    assert torch.equal(
        (source(features) > 0).long(),
        expanded(features))
    trainable = [
        name for name, parameter in expanded.named_parameters()
        if parameter.requires_grad]
    assert trainable == [
        "log_advantage_scale",
        "baseline_value.weight", "baseline_value.bias",
        "third_value.weight", "third_value.bias",
    ]


def test_stored_core_adapter_preserves_old_policy_and_freezes_it() -> None:
    torch.manual_seed(11)
    source = ActionValueHead(8)
    with torch.no_grad():
        source.network[-1].weight.normal_()
        source.network[-1].bias.normal_()
    features = torch.randn(31, 7)
    expanded = StoredCoreThirdValueHead(
        source, input_width=7, hidden=8)
    assert torch.equal(
        source.q_values(features[:, :4]).argmax(-1),
        expanded(features))
    assert all(
        not parameter.requires_grad
        for parameter in expanded.old_values.parameters())
    assert all(
        parameter.requires_grad
        for parameter in expanded.third_residual.parameters())


def test_composed_adapter_uses_champion_order_and_freezes_both_sources() -> None:
    torch.manual_seed(12)
    values = ActionValueHead(8)
    champion = ComputeAdvantageHead(8)
    with torch.no_grad():
        values.network[-1].weight.normal_()
        values.network[-1].bias.normal_()
    features = torch.randn(31, 7)
    composed = ComposedChampionThirdValueHead(
        values, champion, input_width=7, hidden=8)
    assert torch.equal(
        (champion(features[:, :4]) > 0).long(),
        composed(features))
    assert all(
        not parameter.requires_grad
        for parameter in composed.old_values.parameters())
    assert all(
        not parameter.requires_grad
        for parameter in composed.champion.parameters())
    ComposedChampionThirdValueHead,
