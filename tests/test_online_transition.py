from __future__ import annotations

import torch

from neural_computer import (
    ExternalAffineTransitionStatistics,
    ExternalOnlineTransitionContextRouter,
    ExternalRandomFeatureTransitionStatistics,
    ExternalTransitionContextEncoder,
    ExternalTransitionModelBank,
    ExternalTransitionObservation,
)


def _affine_observation(rows: int = 8) -> ExternalTransitionObservation:
    state = torch.arange(rows * 2, dtype=torch.float32).reshape(rows, 2) / 7.0
    intention = torch.arange(rows, dtype=torch.float32).reshape(rows, 1) / 5.0
    features = torch.cat((state, intention, torch.ones(rows, 1)), dim=-1)
    weights = torch.tensor(
        [[1.0, 0.2], [-0.3, 0.8], [0.7, -1.1], [0.4, -0.6]]
    )
    return ExternalTransitionObservation(
        state=state,
        intention=intention,
        next_state=features @ weights,
        confidence=torch.ones(rows),
    )


def test_affine_statistics_is_a_bank_fast_path_without_optimizer_replay() -> None:
    observation = _affine_observation()
    bank = ExternalTransitionModelBank(
        2,
        1,
        3,
        model_family="affine_sufficient_statistics_v1",
        affine_ridge=1e-7,
        capacity=1,
    )
    context = torch.tensor([1.0, 0.0, 0.0])
    bank.ensure_context(context)
    loss_before = bank.adaptation_step(
        observation,
        context.unsqueeze(0).expand(observation.state.shape[0], -1),
        None,
    )
    assert loss_before > 0.0
    assert int(bank.models[0].sample_count) == observation.state.shape[0]
    assert float(bank.loss(observation, context.unsqueeze(0).expand(8, -1))) < 1e-7

    restored = ExternalTransitionModelBank.from_payload(bank.payload())
    assert restored.configuration()["model_family"] == "affine_sufficient_statistics_v1"
    assert restored.content_digest() == bank.content_digest()


def test_nonlinear_bank_can_update_through_memory_boundary_without_optimizer() -> None:
    torch.manual_seed(1302)
    bank = ExternalTransitionModelBank(
        2,
        1,
        3,
        adaptation_learning_rate=0.03,
    )
    context = torch.tensor([1.0, 0.0, 0.0])
    bank.ensure_context(context)
    observation = ExternalTransitionObservation(
        state=torch.randn(4, 2),
        intention=torch.randn(4, 1),
        next_state=torch.randn(4, 2),
    )
    before = bank.models[0].digest()
    bank.adaptation_step(
        observation,
        context.unsqueeze(0).expand(4, -1),
        None,
    )
    assert bank.models[0].digest() != before


def test_random_feature_statistics_learns_nonlinear_transitions_one_pass() -> None:
    for seed in (1401, 1402):
        torch.manual_seed(seed)
        model = ExternalRandomFeatureTransitionStatistics(
            2,
            1,
            feature_width=128,
            ridge=1e-4,
            seed=17,
        )
        state = torch.rand(128, 2) * 2.0 - 1.0
        intention = torch.rand(128, 1) * 2.0 - 1.0
        next_state = torch.cat(
            (
                torch.sin(2.0 * state[:, 0:1] + intention),
                state[:, 0:1] * state[:, 1:2] + intention.square(),
            ),
            dim=-1,
        )
        train = ExternalTransitionObservation(
            state=state[:64],
            intention=intention[:64],
            next_state=next_state[:64],
        )
        heldout = ExternalTransitionObservation(
            state=state[64:],
            intention=intention[64:],
            next_state=next_state[64:],
        )
        model.observe(train)
        assert int(model.sample_count) == 64
        assert float(model.loss(heldout)) < 0.02
        restored = ExternalRandomFeatureTransitionStatistics.from_payload(
            model.state_payload()
        )
        assert restored.digest() == model.digest()
        assert torch.allclose(restored(state[64:], intention[64:]), model(state[64:], intention[64:]))


def test_router_stages_and_promotes_affine_candidate_without_optimizer() -> None:
    bank = ExternalTransitionModelBank(
        2,
        1,
        4,
        model_family="affine_sufficient_statistics_v1",
        affine_ridge=1e-7,
        capacity=1,
    )
    encoder = ExternalTransitionContextEncoder(2, 1, hidden_width=8, context_width=4)
    router = ExternalOnlineTransitionContextRouter(
        bank,
        encoder,
        match_tolerance=1e-8,
        continuation_tolerance=1e9,
        admission_observations=4,
        max_contexts=1,
        defer_admission=True,
    )
    observation = _affine_observation(4)
    rows = [
        ExternalTransitionObservation(
            state=observation.state[row : row + 1],
            intention=observation.intention[row : row + 1],
            next_state=observation.next_state[row : row + 1],
            confidence=torch.ones(1),
        )
        for row in range(4)
    ]
    for row in rows[:3]:
        assert router.observe(row).status == "pending"
    staged = router.observe(rows[3])
    assert staged.status == "staged"
    assert router.adaptation_step(staged, None) > 0.0
    assert int(router.provisional_model_at(0).sample_count) == 4
    restored_router = ExternalOnlineTransitionContextRouter.from_payload(
        router.state_payload()
    )
    assert int(restored_router.provisional_model_at(0).sample_count) == 4

    heldout = _affine_observation(4)
    receipt = router.promote_staged_candidate(
        heldout,
        lambda candidate: candidate.context_count == 1,
        prediction_tolerance=1e-7,
    )
    assert receipt.accepted
    assert router.bank.models[0].sample_count.item() == 4


def test_bank_selects_smallest_verified_model_family_without_mutating_candidates() -> None:
    torch.manual_seed(1303)
    observation = _affine_observation(12)
    heldout = _affine_observation(4)
    bank = ExternalTransitionModelBank(2, 1, 3)
    affine = bank.new_model("affine_sufficient_statistics_v1")
    for row in range(observation.state.shape[0]):
        affine.observe(
            ExternalTransitionObservation(
                state=observation.state[row : row + 1],
                intention=observation.intention[row : row + 1],
                next_state=observation.next_state[row : row + 1],
                confidence=torch.ones(1),
            )
        )
    nonlinear = bank.new_model("nonlinear_mlp_v1")
    affine_digest = affine.digest()
    nonlinear_digest = nonlinear.digest()

    selection = bank.select_model_family_verified(
        {
            "affine_sufficient_statistics_v1": affine,
            "nonlinear_mlp_v1": nonlinear,
        },
        heldout,
        prediction_tolerance=1e-6,
    )
    assert selection.accepted
    assert selection.selected_family == "affine_sufficient_statistics_v1"
    assert affine.digest() == affine_digest
    assert nonlinear.digest() == nonlinear_digest


def test_mixed_bank_keeps_affine_and_nonlinear_slots_independently() -> None:
    torch.manual_seed(1304)
    bank = ExternalTransitionModelBank(
        2,
        1,
        3,
        model_family="mixed_verified_v1",
        capacity=2,
    )
    affine_index = bank.ensure_context(
        torch.tensor([1.0, 0.0, 0.0]),
        model_family="affine_sufficient_statistics_v1",
    )
    nonlinear_index = bank.ensure_context(
        torch.tensor([0.0, 1.0, 0.0]),
        model_family="nonlinear_mlp_v1",
    )
    affine_observation = _affine_observation(8)
    nonlinear_observation = ExternalTransitionObservation(
        state=torch.randn(4, 2),
        intention=torch.randn(4, 1),
        next_state=torch.randn(4, 2),
    )
    observation = ExternalTransitionObservation(
        state=torch.cat((affine_observation.state, nonlinear_observation.state)),
        intention=torch.cat(
            (affine_observation.intention, nonlinear_observation.intention)
        ),
        next_state=torch.cat(
            (affine_observation.next_state, nonlinear_observation.next_state)
        ),
        confidence=torch.ones(12),
    )
    contexts = torch.cat(
        (
            torch.tensor([1.0, 0.0, 0.0]).expand(8, -1),
            torch.tensor([0.0, 1.0, 0.0]).expand(4, -1),
        )
    )
    optimizer = torch.optim.Adam(bank.models[nonlinear_index].parameters(), lr=0.01)
    bank.adaptation_step(observation, contexts, {nonlinear_index: optimizer})
    assert bank.model_family_at(affine_index) == "affine_sufficient_statistics_v1"
    assert bank.model_family_at(nonlinear_index) == "nonlinear_mlp_v1"
    assert int(bank.models[affine_index].sample_count) == 8

    restored = ExternalTransitionModelBank.from_payload(bank.payload())
    assert restored.model_family == "mixed_verified_v1"
    assert restored.model_family_at(affine_index) == "affine_sufficient_statistics_v1"
    assert restored.model_family_at(nonlinear_index) == "nonlinear_mlp_v1"
    assert restored.content_digest() == bank.content_digest()


def test_mixed_router_adapts_both_families_and_promotes_verified_winner() -> None:
    torch.manual_seed(1305)
    bank = ExternalTransitionModelBank(
        2,
        1,
        4,
        model_family="mixed_verified_v1",
        affine_ridge=1e-7,
        capacity=1,
    )
    encoder = ExternalTransitionContextEncoder(2, 1, hidden_width=8, context_width=4)
    router = ExternalOnlineTransitionContextRouter(
        bank,
        encoder,
        match_tolerance=1e-8,
        continuation_tolerance=1e9,
        admission_observations=8,
        max_contexts=1,
        defer_admission=True,
    )
    observation = _affine_observation(8)
    rows = [
        ExternalTransitionObservation(
            state=observation.state[row : row + 1],
            intention=observation.intention[row : row + 1],
            next_state=observation.next_state[row : row + 1],
            confidence=torch.ones(1),
        )
        for row in range(8)
    ]
    for row in rows[:-1]:
        assert router.observe(row).status == "pending"
    staged = router.observe(rows[-1])
    assert staged.status == "staged"
    assert set(router._provisional_candidates[0].models()) == {
        "nonlinear_mlp_v1",
        "affine_sufficient_statistics_v1",
        "random_feature_sufficient_statistics_v1",
    }
    router.adaptation_step(staged, None)
    assert int(
        router._provisional_candidates[0].alternatives[
            "affine_sufficient_statistics_v1"
        ].sample_count
    ) == 8

    restored = ExternalOnlineTransitionContextRouter.from_payload(
        router.state_payload()
    )
    assert set(restored._provisional_candidates[0].models()) == {
        "nonlinear_mlp_v1",
        "affine_sufficient_statistics_v1",
        "random_feature_sufficient_statistics_v1",
    }
    receipt = router.promote_staged_candidate(
        _affine_observation(4),
        lambda candidate: candidate.context_count == 1
        and candidate.model_family_at(0) == "affine_sufficient_statistics_v1",
        prediction_tolerance=1e-6,
    )
    assert receipt.accepted
    assert router.bank.model_family_at(0) == "affine_sufficient_statistics_v1"


def test_affine_transition_statistics_learns_and_persists_one_pass() -> None:
    torch.manual_seed(1301)
    model = ExternalAffineTransitionStatistics(2, 1, ridge=1e-7)
    state = torch.randn(8, 2)
    intention = torch.randn(8, 1)
    features = torch.cat((state, intention, torch.ones(8, 1)), dim=-1)
    true_weights = torch.tensor(
        [
            [1.0, 0.2],
            [-0.3, 0.8],
            [0.7, -1.1],
            [0.4, -0.6],
        ]
    )
    next_state = features @ true_weights
    observation = ExternalTransitionObservation(
        state=state,
        intention=intention,
        next_state=next_state,
        confidence=torch.ones(8),
    )

    for row in range(state.shape[0]):
        model.observe(
            ExternalTransitionObservation(
                state=observation.state[row : row + 1],
                intention=observation.intention[row : row + 1],
                next_state=observation.next_state[row : row + 1],
                confidence=torch.ones(1),
            )
    )

    assert int(model.sample_count) == 8
    assert float(model.loss(observation)) < 1e-7
    heldout_state = torch.randn(4, 2)
    heldout_intention = torch.randn(4, 1)
    heldout_features = torch.cat(
        (heldout_state, heldout_intention, torch.ones(4, 1)), dim=-1
    )
    heldout = ExternalTransitionObservation(
        state=heldout_state,
        intention=heldout_intention,
        next_state=heldout_features @ true_weights,
    )
    assert float(model.loss(heldout)) < 1e-6
    restored = ExternalAffineTransitionStatistics.from_payload(model.state_payload())
    assert restored.digest() == model.digest()
    assert torch.allclose(
        restored(state, intention),
        model(state, intention),
        atol=1e-7,
        rtol=0.0,
    )


def test_affine_transition_statistics_rejects_checksum_corruption() -> None:
    model = ExternalAffineTransitionStatistics(1, 1)
    model.observe(
        ExternalTransitionObservation(
            state=torch.tensor([[1.0]]),
            intention=torch.tensor([[2.0]]),
            next_state=torch.tensor([[3.0]]),
        )
    )
    payload = model.state_payload()
    payload["state"]["normal_matrix"][0, 0] += 1.0

    try:
        ExternalAffineTransitionStatistics.from_payload(payload)
    except ValueError as error:
        assert "checksum" in str(error)
    else:
        raise AssertionError("expected affine statistics checksum rejection")
