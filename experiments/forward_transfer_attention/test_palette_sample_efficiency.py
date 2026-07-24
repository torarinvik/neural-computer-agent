import pytest
import torch

from .probe_palette_sample_efficiency import (
    PaletteInvariantBinder, _balanced_logical_seeds, _balanced_specs,
    _first_sustained_threshold, _palette_for, _parse_pairs)


def test_palette_parser_and_assignment_are_deterministic():
    palettes = _parse_pairs("0,1;1,2;2,3")
    assert palettes == ((0, 1), (1, 2), (2, 3))
    assert _palette_for(123, palettes) == _palette_for(123, palettes)
    assert _palette_for(123, palettes) in palettes


def test_palette_parser_rejects_degenerate_pairs():
    with pytest.raises(ValueError):
        _parse_pairs("0,0")


def test_threshold_requires_two_consecutive_evaluations():
    history = [
        {"step": 20, "examples_seen": 640, "validation_accuracy": 0.61},
        {"step": 40, "examples_seen": 1280, "validation_accuracy": 0.59},
        {"step": 60, "examples_seen": 1920, "validation_accuracy": 0.64},
        {"step": 80, "examples_seen": 2560, "validation_accuracy": 0.65},
    ]
    assert _first_sustained_threshold(history, 0.60) == {
        "step": 60, "examples_seen": 1920}


def test_balanced_specs_balance_every_palette_rule_cell():
    palettes = _parse_pairs("0,1;1,2;2,3")
    specs = _balanced_specs(10_000, 60, palettes, heldout=True)
    counts = {}
    from .environment import _independent_choice
    for seed, palette in specs:
        rule = _independent_choice(
            seed, True, "temporal-atom-rule", 2)
        counts[(palette, rule)] = counts.get((palette, rule), 0) + 1
    assert set(counts.values()) == {10}


def test_balanced_logical_seeds_have_equal_rules():
    from .environment import _independent_choice
    seeds = _balanced_logical_seeds(20_000, 40, heldout=False)
    rules = [
        _independent_choice(seed, False, "temporal-atom-rule", 2)
        for seed in seeds
    ]
    assert rules.count(0) == rules.count(1) == 20


def test_palette_invariant_binder_exposes_stable_relation_latent():
    model = PaletteInvariantBinder(8, width=4)
    snapshots = torch.randn(2, 3, 8)
    repeated = snapshots[:1].expand(2, -1, -1)
    latent = model.relation_latent(repeated)
    assert torch.equal(latent[0], latent[1])
    assert model(repeated).shape == (2, 2)


def test_learning_curve_subsets_balance_every_cell():
    from .environment import _independent_choice
    from .probe_color_object_learning_curve import _nested_balanced_indices
    palettes = _parse_pairs("0,1;1,2;2,3")
    specs = _balanced_specs(41_000_000, 120, palettes, False)
    indices = _nested_balanced_indices(
        41_000_000, 120, palettes, False, 30)
    counts = {}
    for index in indices.tolist():
        seed, palette = specs[index]
        rule = _independent_choice(
            seed, False, "temporal-atom-rule", 2)
        counts[(palette, rule)] = counts.get((palette, rule), 0) + 1
    assert set(counts.values()) == {5}
