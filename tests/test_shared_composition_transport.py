from experiments.external_skill_fragment_composition_amodal.train_shared_multi_target import (
    _group_composition_ids,
    _rotate_orders,
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
