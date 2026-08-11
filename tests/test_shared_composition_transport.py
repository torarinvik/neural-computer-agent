import torch

from experiments.external_skill_fragment_composition_amodal.train import _batch
from experiments.external_skill_fragment_composition_amodal.train_shared_multi_target import (
    _causal_prefix_targets,
    _group_composition_ids,
    _rotate_orders,
    _select_causal_rows,
    _spec_groups_by_length,
)


def test_composition_ids_are_contiguous_per_target():
    assert _group_composition_ids(3, 2).tolist() == [0, 0, 1, 1, 2, 2]


def test_curriculum_groups_only_equal_length_programs():
    specs = (
        ((0, 1), ("a", "b")),
        ((2,), ("c",)),
        ((3, 2), ("d", "e")),
        ((1, 0, 3), ("f", "g", "h")),
    )

    groups = _spec_groups_by_length(specs, max_group_size=2)

    assert [tuple(len(order) for order, _ in group) for group in groups] == [
        (2, 2),
        (1,),
        (3,),
    ]


def test_order_rotation_preserves_atomic_routes_and_changes_composites():
    orders = ((2,), (0, 1, 3), (3, 2))

    assert _rotate_orders(orders) == ((2,), (1, 3, 0), (2, 3))


def test_causal_prefix_targets_reuse_the_same_rendered_sequence():
    programs = (("reverse", "rotate"), ("rotate", "reverse"))
    composition_ids = _group_composition_ids(2, 2)
    batch = _batch(
        operation="generated_composition",
        count=4,
        span=3,
        seed=123,
        generated_compositions=programs,
        generated_composition_ids_override=composition_ids,
    )

    targets = _causal_prefix_targets(
        batch,
        programs,
        composition_ids,
        seed=456,
    )

    assert targets.shape == (4, 2, 3)
    assert torch.all((targets == 0) | (targets == 1))
    assert torch.equal(targets[:, 0], _batch(
        operation="generated_composition",
        count=4,
        span=3,
        seed=999,
        generated_compositions=(("reverse",), ("rotate",)),
        generated_composition_ids_override=composition_ids,
        sequence_override=batch.sequence,
        operation_bits_override=batch.operation_bits,
    ).correct_actions)


def test_active_causal_selection_is_balanced_and_beats_passive_subset():
    programs = (("reverse", "rotate"), ("rotate", "reverse"))
    composition_ids = _group_composition_ids(2, 4)
    batch = _batch(
        operation="generated_composition",
        count=8,
        span=3,
        seed=321,
        generated_compositions=programs,
        generated_composition_ids_override=composition_ids,
    )
    row_scores = torch.tensor([0.1, 0.2, 0.9, 0.8, 0.1, 0.3, 0.7, 0.6])
    causal_signal = row_scores.view(8, 1, 1).expand(8, 3, 2)

    active, active_stats = _select_causal_rows(
        batch,
        causal_signal,
        composition_ids,
        examples_per_target=2,
        mode="active",
        seed=7,
    )
    passive, passive_stats = _select_causal_rows(
        batch,
        causal_signal,
        composition_ids,
        examples_per_target=2,
        mode="passive",
        seed=7,
    )

    assert active.batch_size == passive.batch_size == 4
    assert active_stats["selected_rows"] == passive_stats["selected_rows"] == 4
    assert active_stats["selected_mean_causal_signal"] == 0.75
    assert active_stats["selected_mean_causal_signal"] > passive_stats[
        "selected_mean_causal_signal"
    ]
