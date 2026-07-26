from pathlib import Path

import torch

from .environment import NULL_ACTION, generate_lifetimes
from .memory import DiskLatentMemory
from .model import UnifiedCognitiveController
from .probe_persistent_interface import _add_context_signatures
from .train import attempted_success_loss, evaluate, rollout
from .train_persistent_memory import _grouped_read


def test_lifetime_has_one_correct_action_and_balanced_private_rules() -> None:
    batch = generate_lifetimes(16, 5, seed=11)
    assert batch.frames.shape == (16, 5, 3, 32, 32)
    assert sorted(batch.rule_bits.tolist()) == [0] * 8 + [1] * 8
    assert torch.equal(
        batch.correct_actions,
        batch.stimulus_identities ^ batch.rule_bits.unsqueeze(1))
    assert sorted(batch.stimulus_identities[:, 0].tolist()) == (
        [0] * 8 + [1] * 8)
    supported = generate_lifetimes(
        16, 5, seed=11, support_trials=4)
    assert torch.equal(
        supported.stimulus_identities[:, 1],
        1 - supported.stimulus_identities[:, 0])


def test_rule_counterfactual_changes_answers_not_pixels() -> None:
    normal = generate_lifetimes(8, 4, seed=17)
    reversed_batch = generate_lifetimes(
        8, 4, seed=17, reverse_rules=True)
    assert torch.equal(normal.frames, reversed_batch.frames)
    assert torch.equal(
        normal.correct_actions, 1 - reversed_batch.correct_actions)

    constant = generate_lifetimes(
        8, 4, seed=17, task="constant_action")
    reversed_constant = generate_lifetimes(
        8, 4, seed=17, reverse_rules=True, task="constant_action")
    assert torch.equal(constant.frames, reversed_constant.frames)
    assert torch.equal(
        constant.correct_actions, 1 - reversed_constant.correct_actions)
    assert torch.equal(
        constant.correct_actions,
        constant.rule_bits.unsqueeze(1).expand(-1, 4))

    identity = generate_lifetimes(
        8, 4, seed=17, task="visible_identity")
    reversed_identity = generate_lifetimes(
        8, 4, seed=17, reverse_stimuli=True,
        task="visible_identity")
    assert not torch.equal(identity.frames, reversed_identity.frames)
    assert torch.equal(
        identity.correct_actions, 1 - reversed_identity.correct_actions)


def test_heldout_renderer_changes_public_surface() -> None:
    train = generate_lifetimes(8, 4, seed=19)
    heldout = generate_lifetimes(8, 4, seed=19, heldout=True)
    assert not torch.equal(train.frames, heldout.frames)
    assert torch.equal(train.correct_actions, heldout.correct_actions)


def test_novel_appearances_change_only_public_geometry() -> None:
    bars = generate_lifetimes(
        16, 6, seed=20, task="binary_mapping",
        appearance="bars", support_trials=1)
    for appearance in ("diamonds", "dot_pairs"):
        novel = generate_lifetimes(
            16, 6, seed=20, task="binary_mapping",
            appearance=appearance, support_trials=1)
        assert not torch.equal(bars.frames, novel.frames)
        assert torch.equal(
            bars.stimulus_identities, novel.stimulus_identities)
        assert torch.equal(bars.rule_bits, novel.rule_bits)
        assert torch.equal(bars.correct_actions, novel.correct_actions)
    diamonds = generate_lifetimes(
        16, 6, seed=20, task="binary_mapping",
        appearance="diamonds", support_trials=1)
    reversed_diamonds = generate_lifetimes(
        16, 6, seed=20, task="binary_mapping",
        appearance="diamonds", support_trials=1,
        reverse_rules=True)
    assert torch.equal(diamonds.frames, reversed_diamonds.frames)
    assert torch.equal(
        diamonds.correct_actions,
        1 - reversed_diamonds.correct_actions)


def test_four_rule_support_is_identifiable_and_balanced() -> None:
    batch = generate_lifetimes(
        16, 6, seed=21, task="four_rule", support_trials=2)
    assert sorted(batch.rule_bits.tolist()) == [0] * 4 + [1] * 4 + (
        [2] * 4 + [3] * 4)
    assert torch.equal(
        batch.stimulus_identities[:, 1],
        1 - batch.stimulus_identities[:, 0])
    rules = batch.rule_bits.unsqueeze(1).expand(-1, batch.trials)
    expected = torch.where(
        rules < 2, rules,
        batch.stimulus_identities ^ (rules - 2))
    assert torch.equal(batch.correct_actions, expected)
    reversed_batch = generate_lifetimes(
        16, 6, seed=21, task="four_rule", support_trials=2,
        reverse_rules=True)
    assert torch.equal(batch.frames, reversed_batch.frames)
    assert torch.equal(
        batch.correct_actions, 1 - reversed_batch.correct_actions)


def test_hidden_rule_gate_requires_real_vision(monkeypatch) -> None:
    """A feedback-only shortcut must never be admitted as composition."""
    model = UnifiedCognitiveController()

    def fake_rollout(_model, batch, **_kwargs):
        # Deliberately use verifier answers regardless of the input frame.
        actions = batch.correct_actions.clone()
        actions[:, 0] = 0
        return {
            "actions": actions,
            "rewards": (
                actions == batch.correct_actions).to(torch.float32),
            "logits": torch.zeros(batch.batch_size, batch.trials, 2),
            "final_workspace": torch.zeros(
                batch.batch_size, model.workspace_slots, model.width),
            "final_hidden": torch.zeros(batch.batch_size, model.width),
        }

    monkeypatch.setattr(
        "experiments.unified_cognitive_controller.train.rollout",
        fake_rollout)
    report = evaluate(
        model, count=8, trials=6, seed=42, device=torch.device("cpu"),
        task="four_rule", feedback_trials=2)
    assert not report["gate"]["vision_causally_used"]
    assert not report["gate"]["accepted"]


def test_unified_controller_rollout_and_workspace_shapes() -> None:
    model = UnifiedCognitiveController(
        width=32, workspace_slots=4, intention_width=8)
    batch = generate_lifetimes(4, 3, seed=23)
    result = rollout(model, batch, sample_actions=False)
    assert result["actions"].shape == (4, 3)
    assert result["logits"].shape == (4, 3, 2)
    assert result["final_workspace"].shape == (4, 4, 32)
    assert result["final_hidden"].shape == (4, 32)

    no_feedback = rollout(
        model, batch, sample_actions=False, feedback_trials=0)
    assert no_feedback["actions"].shape == (4, 3)


def test_attempted_loss_has_no_unattempted_target_argument() -> None:
    logits = torch.tensor([[0.2, -0.4], [0.3, 0.8]], requires_grad=True)
    actions = torch.tensor([0, 1])
    outcomes = torch.tensor([1.0, 0.0])
    loss = attempted_success_loss(logits, actions, outcomes)
    loss.backward()
    assert logits.grad is not None
    assert logits.grad[0, 1] == 0
    assert logits.grad[1, 0] == 0


def test_disk_latent_memory_round_trip(tmp_path: Path) -> None:
    memory = DiskLatentMemory(width=8, capacity=2)
    keys = torch.eye(8)[:2]
    values = torch.arange(16, dtype=torch.float32).reshape(2, 8)
    assert memory.commit(
        keys, values, torch.tensor([0.9, 0.1]), threshold=0.5) == 1
    path = tmp_path / "memory.pt"
    memory.save(path)
    restored = DiskLatentMemory.load(path)
    assert restored.count == 1
    read, confidence = restored.retrieve(keys[:1], top_k=1)
    assert torch.allclose(read, values[:1])
    assert confidence.shape == (1,)


def test_disk_memory_can_report_cosine_match_confidence() -> None:
    memory = DiskLatentMemory(width=4, capacity=2)
    keys = torch.eye(4)[:2]
    values = torch.arange(8, dtype=torch.float32).reshape(2, 4)
    strengths = torch.tensor([0.5, 0.9])
    assert memory.commit(
        keys, values, strengths, threshold=0.0) == 2
    _, ranked = memory.retrieve(keys[:1], top_k=1)
    read, cosine = memory.retrieve(
        keys[:1], top_k=1, confidence_mode="cosine")
    assert torch.allclose(read, values[:1])
    assert torch.allclose(cosine, torch.ones_like(cosine))
    assert ranked.item() < cosine.item()


def test_disk_memory_reports_adaptive_read_features() -> None:
    memory = DiskLatentMemory(width=4, capacity=2)
    keys = torch.eye(4)[:2]
    values = torch.arange(8, dtype=torch.float32).reshape(2, 4)
    strengths = torch.tensor([0.5, 0.9])
    assert memory.commit(
        keys, values, strengths, threshold=0.0) == 2
    read, features = memory.retrieve_with_features(keys[:1])
    assert torch.allclose(read, values[:1])
    assert features.shape == (1, 4)
    assert torch.allclose(features[:, 0], torch.ones(1))
    assert torch.allclose(features[:, 2], strengths[:1])
    assert torch.allclose(features[:, 3], torch.ones(1))


def test_adaptive_memory_read_is_optional_and_bounded() -> None:
    ordinary = UnifiedCognitiveController(
        width=32, workspace_slots=4, intention_width=8)
    assert ordinary.memory_read_gate is None
    model = UnifiedCognitiveController(
        width=32, workspace_slots=4, intention_width=8,
        adaptive_memory_read=True)
    probability = model.memory_read_probability(torch.zeros(3, 4))
    assert probability.shape == (3,)
    assert torch.all((probability > 0) & (probability < 1))
    nonlinear = UnifiedCognitiveController(
        width=32, workspace_slots=4, intention_width=8,
        adaptive_memory_read=True, adaptive_memory_read_hidden=8)
    assert sum(
        parameter.numel()
        for parameter in nonlinear.memory_read_gate.parameters()) == 49


def test_recurring_context_signature_is_stable_within_world() -> None:
    batch = generate_lifetimes(
        8, 3, seed=31, task="binary_mapping", support_trials=1)
    marked = _add_context_signatures(batch, seed=41)
    signatures = marked.frames[:, :, :, 2:5, 2:5]
    assert torch.equal(signatures[:, 0], signatures[:, 1])
    assert torch.equal(signatures[:, 1], signatures[:, 2])
    assert not torch.equal(signatures[0, 0], signatures[1, 0])


def test_grouped_memory_read_never_crosses_memory_banks() -> None:
    keys = torch.tensor([
        [1.0, 0.0], [0.0, 1.0],
        [1.0, 0.0], [0.0, 1.0],
    ])
    values = torch.tensor([
        [1.0, 10.0], [2.0, 20.0],
        [3.0, 30.0], [4.0, 40.0],
    ])
    read, selected = _grouped_read(
        keys, keys, values, capacity=2, mode="hard")
    assert torch.equal(read, values)
    assert torch.equal(selected, torch.tensor([0, 1, 0, 1]))
