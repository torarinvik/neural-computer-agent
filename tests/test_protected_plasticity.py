import torch

from experiments.games_amodal.protected_plasticity import (
    controller_named_parameters,
    random_reference_like,
    train_phase,
)
from experiments.games_amodal.shared_controller import (
    SharedControllerAgent,
    set_trainable,
)


def _agent() -> SharedControllerAgent:
    torch.manual_seed(0)
    return SharedControllerAgent(
        event_width=32, intention_width=16, feedback_width=8, hidden=16
    )


def test_reference_accumulates_only_from_requested_update() -> None:
    agent = _agent()
    _, reference = train_phase(
        agent,
        "snake",
        updates=3,
        batch_size=4,
        steps=8,
        seed=1,
        gamma=0.9,
        learning_rate=1e-3,
        accumulate_reference_from=1,
    )
    assert reference is not None
    total = sum(float(tensor.abs().sum()) for tensor in reference.values())
    assert total > 0.0
    assert set(reference) == {
        name for name, _ in controller_named_parameters(agent)
    }


def test_random_reference_matches_global_norm() -> None:
    agent = _agent()
    _, reference = train_phase(
        agent,
        "snake",
        updates=2,
        batch_size=4,
        steps=8,
        seed=2,
        gamma=0.9,
        learning_rate=1e-3,
        accumulate_reference_from=0,
    )
    assert reference is not None
    random_map = random_reference_like(reference, seed=3)
    norm = lambda m: float(
        torch.sqrt(sum(t.square().sum() for t in m.values()))
    )
    assert abs(norm(random_map) - norm(reference)) < 1e-4 * max(norm(reference), 1.0)
    cosine = sum(
        float((reference[name] * random_map[name]).sum()) for name in reference
    ) / max(norm(reference) * norm(random_map), 1e-12)
    assert abs(cosine) < 0.2


def test_projection_keeps_core_plastic_but_records_engagement() -> None:
    agent = _agent()
    _, reference = train_phase(
        agent,
        "snake",
        updates=2,
        batch_size=4,
        steps=8,
        seed=4,
        gamma=0.9,
        learning_rate=1e-3,
        accumulate_reference_from=0,
    )
    assert reference is not None
    core_before = [p.detach().clone() for p in agent.controller.parameters()]
    set_trainable(agent.game_modules("snake"), False)
    history, _ = train_phase(
        agent,
        "pong",
        updates=2,
        batch_size=4,
        steps=8,
        seed=5,
        gamma=0.9,
        learning_rate=1e-3,
        reference=reference,
    )
    changed = any(
        not torch.equal(before, after)
        for before, after in zip(
            core_before, agent.controller.parameters(), strict=True
        )
    )
    assert changed
    assert all("gradient_projected" in entry for entry in history)
    assert all(entry["replayed_examples"] == 0.0 for entry in history)
