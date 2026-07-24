import numpy as np
import pytest

from .environment import (SHOTS, generate_attention_lifetime,
                          generate_compositional_temporal_attention_lifetime,
                          generate_shape_attention_lifetime,
                          generate_temporal_first_lifetime,
                          generate_temporal_attention_lifetime,
                          generate_temporal_grounding_lifetime,
                          generate_temporal_last_lifetime)


def test_attention_lifetime_is_deterministic_and_sensory_only():
    first = generate_attention_lifetime(17)
    repeated = generate_attention_lifetime(17)
    assert first.rule == repeated.rule
    assert first.cue_code == repeated.cue_code
    assert first.color_mapping == repeated.color_mapping
    assert np.array_equal(first.studies[0].frames, repeated.studies[0].frames)
    assert first.shots == SHOTS
    assert len(first.supports) == max(SHOTS)


def test_every_future_query_has_one_verifiable_answer():
    for seed in range(64):
        lifetime = generate_attention_lifetime(seed, query_count=4)
        assert len(lifetime.future_queries) == 4
        for episode, (color, shape) in zip(lifetime.future_queries,
                                           lifetime.query_features):
            expected = lifetime.color_mapping[(color, shape)[lifetime.rule]]
            assert episode.actions.tolist() == [expected]
            assert 0 <= expected < 8


def test_support_always_disambiguates_the_attention_rule():
    for seed in range(128):
        lifetime = generate_attention_lifetime(seed)
        for (color, shape), answer, support in zip(
                lifetime.support_features, lifetime.support_answers, lifetime.supports):
            color_answer = lifetime.color_mapping[color]
            shape_answer = lifetime.color_mapping[shape]
            assert color_answer != shape_answer
            assert answer == (color_answer, shape_answer)[lifetime.rule]
            # The correct action is rendered as feedback, never passed as an
            # environmental input tensor to the controller.
            assert support.actions.tolist() == [1]


def test_invalid_query_count_is_rejected():
    with pytest.raises(ValueError):
        generate_attention_lifetime(0, query_count=0)


def test_cue_pixels_do_not_reveal_rule_through_prng_order():
    lifetimes = [generate_attention_lifetime(seed) for seed in range(4096)]
    # If cue parity or its high bit predicts the rule, a network can bypass
    # demonstrations. Independently hashed assignments should remain balanced.
    for feature in (lambda code: code & 1, lambda code: (code >> 6) & 1):
        rates = []
        for value in (0, 1):
            group = [item.rule for item in lifetimes if feature(item.cue_code) == value]
            rates.append(sum(group) / len(group))
        assert abs(rates[0] - rates[1]) < 0.05


def test_shape_attention_is_deterministic_and_uniquely_scored():
    first = generate_shape_attention_lifetime(71)
    repeated = generate_shape_attention_lifetime(71)
    assert first.rule == repeated.rule
    assert first.support_answers == repeated.support_answers
    assert np.array_equal(first.supports[0].frames, repeated.supports[0].frames)
    assert len(set(first.color_mapping)) == 2
    assert first.future_queries[0].actions[0] in first.color_mapping


def test_shape_cue_pixels_do_not_reveal_target_shape():
    lifetimes = [generate_shape_attention_lifetime(seed) for seed in range(4096)]
    for feature in (lambda code: code & 1, lambda code: (code >> 6) & 1):
        rates = []
        for value in (0, 1):
            group = [item.rule for item in lifetimes if feature(item.cue_code) == value]
            rates.append(sum(group) / len(group))
        assert abs(rates[0] - rates[1]) < 0.05


def test_temporal_attention_is_deterministic_and_order_is_causal():
    first = generate_temporal_attention_lifetime(93)
    repeated = generate_temporal_attention_lifetime(93)
    assert first.rule == repeated.rule
    assert first.support_answers == repeated.support_answers
    assert np.array_equal(first.supports[0].frames, repeated.supports[0].frames)
    assert first.supports[0].length == 3
    assert first.future_queries[0].length == 2
    assert np.count_nonzero(first.future_queries[0].pcm) == 0
    assert len(first.color_mapping) == 2
    # Queries alternate orientation, so no constant selected colour/response
    # can solve the future set.
    assert len({episode.actions[0] for episode in first.future_queries}) == 2
    # The same identity is pixel-identical across first/last positions and
    # across query slots; only stream order can distinguish a reversal.
    rendered = {}
    for order, episode in zip(first.query_features, first.future_queries):
        for color, frame in zip(order, episode.frames):
            if color in rendered:
                assert np.array_equal(frame, rendered[color])
            else:
                rendered[color] = frame
    # Temporal position must not be watermarked by circle-vs-square pixels: an
    # object has the same appearance when it occurs first or last.
    color_frames = {}
    for order, episode in zip(first.query_features, first.future_queries):
        for color, frame in zip(order, episode.frames):
            crop = frame[36:69, 36:125]
            if color in color_frames:
                # Background varies, but the non-background object mask and
                # geometry are position-invariant.
                previous = color_frames[color]
                assert np.count_nonzero(crop[..., color] > 120) == previous
            else:
                color_frames[color] = np.count_nonzero(crop[..., color] > 120)
    for order, episode in zip(first.query_features, first.future_queries):
        expected = first.color_mapping[order[first.rule]]
        reversed_answer = first.color_mapping[tuple(reversed(order))[first.rule]]
        assert episode.actions.tolist() == [expected, expected]
        assert expected != reversed_answer


def test_temporal_feedback_diagnostic_variants_preserve_labels_but_change_pixels():
    original = generate_temporal_attention_lifetime(93)
    fat = generate_temporal_attention_lifetime(93, mapping_line_width=12)
    direct = generate_temporal_attention_lifetime(93, feedback_mode="color-button")
    object_feedback = generate_temporal_attention_lifetime(
        93, feedback_mode="color-object")
    assert original.rule == fat.rule == direct.rule == object_feedback.rule
    assert (original.support_answers == fat.support_answers ==
            direct.support_answers == object_feedback.support_answers)
    assert not np.array_equal(original.studies[0].frames, fat.studies[0].frames)
    assert not np.array_equal(original.supports[0].frames[-1],
                              direct.supports[0].frames[-1])
    assert not np.array_equal(direct.supports[0].frames[-1],
                              object_feedback.supports[0].frames[-1])


def test_temporal_palette_substitution_preserves_logic_but_changes_identity_pixels():
    original = generate_temporal_attention_lifetime(
        93, feedback_mode="color-button")
    substituted = generate_temporal_attention_lifetime(
        93, feedback_mode="color-button", color_ids=(2, 3))
    assert original.rule == substituted.rule
    assert original.color_mapping == substituted.color_mapping
    assert original.support_features == substituted.support_features
    assert original.support_answers == substituted.support_answers
    assert original.query_features == substituted.query_features
    assert not np.array_equal(original.studies[0].frames,
                              substituted.studies[0].frames)
    assert not np.array_equal(original.supports[0].frames,
                              substituted.supports[0].frames)


def test_temporal_render_seed_changes_nuisance_pixels_not_task_metadata():
    original = generate_temporal_attention_lifetime(93, render_seed=1_000_093)
    augmented = generate_temporal_attention_lifetime(93, render_seed=2_000_093)
    assert original.rule == augmented.rule
    assert original.color_mapping == augmented.color_mapping
    assert original.support_features == augmented.support_features
    assert original.support_answers == augmented.support_answers
    assert original.query_features == augmented.query_features
    assert not np.array_equal(original.studies[0].frames,
                              augmented.studies[0].frames)


def test_event_snapshot_probe_splits_logical_lifetimes_before_augmentation():
    from .probe_temporal_event_snapshot_binder import (
        TEST_LOGICAL_START,
        TRAIN_LOGICAL_START,
        _assert_disjoint_logical_splits,
        _logical_lifetime_ids,
    )

    train_ids = _logical_lifetime_ids(TRAIN_LOGICAL_START, 16_384)
    test_ids = _logical_lifetime_ids(TEST_LOGICAL_START, 16_384)
    assert train_ids.isdisjoint(test_ids)
    _assert_disjoint_logical_splits(16_384, 16_384)


def test_event_snapshot_cache_subsets_keep_variants_grouped_by_lifetime():
    import torch

    from .probe_temporal_event_snapshot_binder import _subset_rendered_lifetimes

    values = torch.arange(4 * 3)
    (selected,), lifetimes, variants = _subset_rendered_lifetimes(
        (values,), cached_lifetimes=4, cached_variants=3,
        subset_lifetimes=2, subset_variants=2)
    assert selected.tolist() == [0, 1, 3, 4]
    assert (lifetimes, variants) == (2, 2)


def test_temporal_counterfactual_reverses_objects_but_not_feedback():
    from .probe_temporal_event_snapshot_binder import _reverse_episode_events

    lifetime = generate_temporal_attention_lifetime(
        93, heldout=True, feedback_mode="color-button")
    original = lifetime.supports[0]
    reversed_episode = _reverse_episode_events(original)
    assert np.array_equal(reversed_episode.frames[0], original.frames[1])
    assert np.array_equal(reversed_episode.frames[1], original.frames[0])
    assert np.array_equal(reversed_episode.frames[2], original.frames[2])
    assert lifetime.support_features[0][lifetime.rule] == (
        lifetime.support_features[0][::-1][1 - lifetime.rule])


def test_temporal_training_augmentation_adds_correct_reversed_queries():
    from .train_joint_adapter import _add_temporal_counterfactual_queries

    lifetime = generate_temporal_attention_lifetime(93, query_count=4)
    augmented = _add_temporal_counterfactual_queries(lifetime)
    assert all(left is right for left, right in
               zip(augmented.future_queries[:4], lifetime.future_queries))
    assert len(augmented.future_queries) == 8
    for order, original, reversed_episode in zip(
            lifetime.query_features, lifetime.future_queries,
            augmented.future_queries[4:]):
        assert np.array_equal(reversed_episode.frames, original.frames[::-1])
        expected = lifetime.color_mapping[order[1 - lifetime.rule]]
        assert reversed_episode.actions.tolist() == [expected] * len(original.actions)


def test_compositional_temporal_level_retains_four_identity_challenge():
    lifetime = generate_compositional_temporal_attention_lifetime(93)
    assert len(lifetime.color_mapping) == 4
    assert len({episode.actions[0] for episode in lifetime.future_queries}) >= 3


def test_temporal_cue_pixels_do_not_reveal_first_or_last_rule():
    lifetimes = [generate_temporal_attention_lifetime(seed) for seed in range(4096)]
    for feature in (lambda code: code & 1, lambda code: (code >> 6) & 1):
        rates = []
        for value in (0, 1):
            group = [item.rule for item in lifetimes if feature(item.cue_code) == value]
            rates.append(sum(group) / len(group))
        assert abs(rates[0] - rates[1]) < 0.05


def test_grounding_curriculum_has_stable_but_visual_only_rule_cues():
    lifetimes = [generate_temporal_grounding_lifetime(seed) for seed in range(64)]
    assert {item.cue_code for item in lifetimes} == {19, 108}
    assert all(item.cue_code == (19, 108)[item.rule] for item in lifetimes)
    assert all(np.count_nonzero(item.future_queries[0].pcm) == 0 for item in lifetimes)


def test_unary_temporal_curriculum_breaks_first_last_symmetry():
    assert all(generate_temporal_first_lifetime(seed).rule == 0 for seed in range(32))
    assert all(generate_temporal_last_lifetime(seed).rule == 1 for seed in range(32))
