import pytest
import torch

from neural_computer import (
    ExternalAgentBrainBank,
    ExternalExecutiveSkillRouter,
    build_temporal_equality_executive_artifact,
)


def test_executive_skill_router_selects_opaque_context_preferences_and_logs_state() -> None:
    bank = ExternalAgentBrainBank(controller_digest="0" * 64, capacity=4)
    bank.admit_executive(
        build_temporal_equality_executive_artifact(event_width=3, delay=1),
        [1.0],
    )
    bank.admit_executive(
        build_temporal_equality_executive_artifact(event_width=3, delay=2),
        [1.0],
    )
    router = ExternalExecutiveSkillRouter(
        bank,
        context_width=3,
        min_mastery_observations=2,
    )
    context_a = torch.tensor([1.0, 0.0, 0.0])
    context_b = torch.tensor([0.0, 1.0, 0.0])
    first = router.select(context_a)
    assert first.slot == 0
    assert first.artifact.digest() == bank.artifact("executive_program", 0).digest()
    for _ in range(2):
        bank.observe_executive_route(context_a, 1, 1.0)
        bank.observe_executive_route(context_b, 0, 1.0)

    selected_a = router.select(context_a)
    selected_b = router.select(context_b)
    assert (selected_a.slot, selected_b.slot) == (1, 0)
    assert selected_a.propensity == pytest.approx(1.0)
    assert selected_b.propensity == pytest.approx(1.0)
    payload = router.state_payload()
    assert payload["bank_digest"] == bank.digest()


def test_executive_skill_router_rejects_stale_artifact_feedback() -> None:
    bank = ExternalAgentBrainBank(controller_digest="1" * 64, capacity=2)
    artifact = build_temporal_equality_executive_artifact(event_width=3, delay=1)
    bank.admit_executive(artifact, [1.0])
    router = ExternalExecutiveSkillRouter(bank, context_width=3)
    selection = router.select(torch.tensor([1.0, 0.0, 0.0]))
    tampered = type(selection)(
        slot=selection.slot,
        propensity=selection.propensity,
        context=selection.context,
        artifact=build_temporal_equality_executive_artifact(event_width=3, delay=2),
        bank_digest=selection.bank_digest,
        bank_version=selection.bank_version,
    )
    with pytest.raises(ValueError, match="no longer matches"):
        router.observe(tampered, 1.0)
