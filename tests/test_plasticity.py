import torch

from neural_computer import ExternalMemoryWritePolicy, MemoryWriteObservation


def _observation(batch: int = 3) -> MemoryWriteObservation:
    return MemoryWriteObservation(
        event=torch.randn(batch, 4),
        hidden=torch.randn(batch, 4),
        workspace_read=torch.randn(batch, 4),
        query_key=torch.randn(batch, 4),
        write_value=torch.randn(batch, 4),
        controller_write_proposal=torch.rand(batch),
        controller_write_context=torch.randn(batch, 8),
        controller_write_relevance=torch.rand(batch),
        memory_read_value=torch.randn(batch, 4),
        memory_read_hit=torch.tensor([True, False, True])[:batch],
        action=torch.randn(batch, 2),
        reward=torch.randn(batch),
        propensity=torch.rand(batch),
        has_feedback=torch.ones(batch),
    )


def test_external_memory_writer_has_stable_opaque_boundary() -> None:
    policy = ExternalMemoryWritePolicy(
        event_width=4,
        hidden_width=4,
        workspace_width=4,
        key_width=4,
        value_width=4,
        memory_read_width=4,
        action_width=2,
        controller_write_context_width=8,
        controller_write_relevance_width=1,
    )

    observation = _observation()
    probability = policy(observation)

    assert probability.shape == (3,)
    assert probability.shape == (3,)
    assert bool(torch.all((probability >= 0.0) & (probability <= 1.0)))
    assert policy.configuration()["schema"] == (
        "neural-computer.external-memory-write-policy.v9"
    )


def test_external_memory_writer_can_train_without_controller_parameters() -> None:
    policy = ExternalMemoryWritePolicy(
        event_width=4,
        hidden_width=4,
        workspace_width=4,
        key_width=4,
        value_width=4,
        memory_read_width=4,
        action_width=2,
        controller_write_context_width=8,
        controller_write_relevance_width=1,
    )
    optimizer = torch.optim.SGD(policy.parameters(), lr=0.1)
    before = [parameter.detach().clone() for parameter in policy.parameters()]

    loss = -torch.log(policy(_observation())).mean()
    loss.backward()
    optimizer.step()

    assert any(
        not torch.equal(old, new)
        for old, new in zip(before, policy.parameters(), strict=True)
    )


def test_external_memory_writer_rejects_nonfinite_observations() -> None:
    policy = ExternalMemoryWritePolicy(
        event_width=4,
        hidden_width=4,
        workspace_width=4,
        key_width=4,
        value_width=4,
        memory_read_width=4,
        action_width=2,
        controller_write_context_width=8,
        controller_write_relevance_width=1,
    )
    observation = _observation()
    observation = MemoryWriteObservation(
        **{
            **observation.__dict__,
            "event": torch.full((3, 4), float("nan")),
        }
    )

    try:
        policy(observation)
    except ValueError as error:
        assert "non-finite" in str(error)
    else:
        raise AssertionError("non-finite memory observations must be rejected")
