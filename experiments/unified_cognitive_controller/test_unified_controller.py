from pathlib import Path

import pytest
import torch

from .environment import NULL_ACTION, generate_lifetimes
from .memory import DiskLatentMemory, TieredLatentMemory
from .model import UnifiedCognitiveController, full_memory_usage_features
from .probe_persistent_interface import _add_context_signatures
from .train import attempted_success_loss, evaluate, rollout
from .train_frequency_recency_replacement import frequency_recency_batch
from .train_memory_replacement import _bank_outcomes
from .train_redundancy_transfer import (
    _expanded_eight_feature_controller,
    _stable_crossing,
    build_transfer_arms,
    initialize_from_saved_strategy,
    redundancy_utility_batch,
)
from .train_contextual_full_residual import (
    adapter_scores,
    full_residual_context_key,
    residual_scores,
    retrieve_residual,
    select_verified_candidate,
)
from .train_passive_replacement_critic import (
    CRITIC_INPUT_WIDTH,
    apply_evidence_control,
    attempted_action_features,
    concordance,
    critic_evidence,
    exploration_probabilities,
    immediate_read_evidence,
)
from .train_shadow_compute_critic import (
    ShadowComputeCritic,
    _shadow_metrics,
    controlled_features,
    selected_compute_loss,
)
from .train_shadow_compute_advantage import (
    ComputeAdvantageHead,
    attempted_advantage_target,
)
from .train_thought_compute_transfer import (
    _active_features as active_thought_features,
    _metrics as thought_compute_metrics,
    _stable_bits as stable_thought_bits,
)
from .train_fourth_primitive_transfer import (
    _alignment_volatility,
    _blend_unit_update,
    _gradient_alignment,
    _importance_volatility,
    _prior_slot_prefixes,
    _relative_state_drift,
)
from .train_shared_compute_value import (
    SharedComputeValue,
    initialize_from_advantage,
)
from .train_cost_aware_requery import (
    CostAwareComputeValue,
    initialize_from_four_feature,
)
from .train_requery_transfer import _candidate_name
from .train_safe_requery_adaptation import (
    ActionValueHead,
    cross_fitted_action_values,
    cross_fitted_context_baseline,
    disagreement_indices,
    head_from_skill_payload,
    paired_ips_improvement,
    skill_head_payload,
)
from .verified_skill_store import VerifiedSkillStore
from .train_persistent_memory import _grouped_read
from .probe_persistent_physical_stream import (
    _future_for_actions,
    _ranked_age,
)
from .compare_persistent_fresh_efficiency import _arm_metrics
from .strategy_memory import (
    LatentStrategyMemory,
    VerifierTrainedContextEncoder,
    physical_context_key,
)
from .train_persistent_physical_utility_adaptation import (
    _curriculum_phases,
    _restore_strategy_memory,
    _strategy_memory_payload,
    _value_diverse_admission,
)
from .dynamic_working_memory import (
    CapabilityLedger,
    DynamicWorkingMemory,
    LatencyTimer,
)


def test_requery_curriculum_preserves_operation_aligned_head() -> None:
    assert _candidate_name("trunk") == "inherited_trunk"
    assert _candidate_name("full") == "inherited"
    with pytest.raises(ValueError):
        _candidate_name("unknown")


def test_paired_ips_promotion_requires_verified_policy_difference() -> None:
    attempted = torch.tensor([0, 1] * 128)
    utilities = attempted.float()
    incumbent = torch.zeros_like(attempted)
    challenger = torch.ones_like(attempted)
    evidence = paired_ips_improvement(
        incumbent, challenger, attempted, utilities)
    assert evidence["lower_95"] > 0
    identical = paired_ips_improvement(
        incumbent, incumbent, attempted, utilities)
    assert identical["estimated_improvement"] == 0
    assert identical["lower_95"] == 0


def test_cross_fitted_context_baseline_uses_no_heldout_outcome() -> None:
    features = torch.arange(32, dtype=torch.float32).reshape(8, 4)
    outcomes = features[:, 0] * 0.1 + 0.5
    original = cross_fitted_context_baseline(features, outcomes)
    changed = outcomes.clone()
    changed[0] += 100
    perturbed = cross_fitted_context_baseline(features, changed)
    assert torch.equal(original[0], perturbed[0])


def test_cross_fitted_action_values_do_not_use_heldout_outcome() -> None:
    features = torch.arange(32, dtype=torch.float32).reshape(8, 4)
    actions = torch.tensor([0, 1] * 4)
    outcomes = features[:, 0] * 0.1 + actions.float() * 0.2
    original = cross_fitted_action_values(features, actions, outcomes)
    changed = outcomes.clone()
    changed[0] += 100
    perturbed = cross_fitted_action_values(features, actions, changed)
    assert torch.equal(original[0][0], perturbed[0][0])
    assert torch.equal(original[1][0], perturbed[1][0])
    evidence = paired_ips_improvement(
        torch.zeros_like(actions), torch.ones_like(actions),
        actions, outcomes, features,
        baseline_mode="doubly_robust_crossfit")
    assert torch.isfinite(torch.tensor(evidence["lower_95"]))


def test_verified_skill_store_is_atomic_hash_checked_and_append_only(
        tmp_path: Path) -> None:
    store = VerifiedSkillStore(tmp_path / "skills")
    parent = store.commit(
        {"weights": torch.tensor([1.0])},
        context_key=torch.tensor([1.0, 0.0]),
        lower_confidence_bound=0.1, verifier_bits=10,
        parent_id=None, provenance={"kind": "audit"})
    child = store.commit(
        {"weights": torch.tensor([2.0])},
        context_key=torch.tensor([0.0, 1.0]),
        lower_confidence_bound=0.2, verifier_bits=20,
        parent_id=parent, provenance={"kind": "promotion"})
    assert torch.equal(
        store.load(parent)["payload"]["weights"], torch.tensor([1.0]))
    assert len(store.entries()) == 2
    child_entry = next(
        row for row in store.entries() if row["skill_id"] == child)
    child_path = store.root / child_entry["file"]
    child_path.write_bytes(child_path.read_bytes() + b"corrupt")
    with pytest.raises(ValueError, match="SHA-256"):
        store.load(child)
    assert torch.equal(
        store.load(parent)["payload"]["weights"], torch.tensor([1.0]))
    with pytest.raises(ValueError, match="unverified"):
        store.commit(
            {"weights": torch.tensor([3.0])},
            context_key=torch.tensor([1.0, 1.0]),
            lower_confidence_bound=0.0, verifier_bits=1,
            parent_id=parent, provenance={})


def test_active_selection_prefers_policy_disagreements() -> None:
    incumbent = ComputeAdvantageHead(4)
    candidate = ComputeAdvantageHead(4)
    with torch.no_grad():
        incumbent.network[-1].bias.fill_(-1)
        candidate.network[-1].bias.fill_(1)
    features = torch.randn(12, 4)
    selected, fraction = disagreement_indices(
        incumbent, candidate, features, count=5)
    assert fraction == 1.0
    assert selected.tolist() == [0, 1, 2, 3, 4]


def test_action_value_head_emits_value_difference() -> None:
    head = ActionValueHead(4)
    features = torch.randn(6, 4)
    values = head.q_values(features)
    assert values.shape == (6, 2)
    assert torch.equal(head(features), values[:, 1] - values[:, 0])
    restored = head_from_skill_payload(
        skill_head_payload(head), torch.device("cpu"))
    assert isinstance(restored, ActionValueHead)
    assert torch.equal(head(features), restored(features))


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


def test_contextual_mapping_requires_two_independent_supports() -> None:
    batch = generate_lifetimes(
        16, 6, seed=22, task="contextual_mapping", support_trials=2)
    assert batch.context_ids is not None
    assert sorted(batch.rule_bits.tolist()) == [0] * 4 + [1] * 4 + (
        [2] * 4 + [3] * 4)
    assert torch.equal(batch.context_ids[:, 0], torch.zeros(16, dtype=torch.long))
    assert torch.equal(batch.context_ids[:, 1], torch.ones(16, dtype=torch.long))
    expected = batch.stimulus_identities ^ (
        (batch.rule_bits.unsqueeze(1) >> batch.context_ids) & 1)
    assert torch.equal(batch.correct_actions, expected)

    reversed_batch = generate_lifetimes(
        16, 6, seed=22, task="contextual_mapping", support_trials=2,
        reverse_rules=True)
    assert torch.equal(batch.frames, reversed_batch.frames)
    assert torch.equal(batch.context_ids, reversed_batch.context_ids)
    assert torch.equal(
        batch.correct_actions, 1 - reversed_batch.correct_actions)


def test_contextual_override_is_a_strictly_easier_context_rung() -> None:
    batch = generate_lifetimes(
        16, 6, seed=23, task="contextual_override", support_trials=2)
    assert batch.context_ids is not None
    assert torch.equal(batch.context_ids[:, 0], torch.zeros(16, dtype=torch.long))
    assert torch.equal(batch.context_ids[:, 1], torch.ones(16, dtype=torch.long))
    mapping = batch.stimulus_identities ^ batch.rule_bits.unsqueeze(1)
    expected = torch.where(
        batch.context_ids == 0, mapping, torch.zeros_like(mapping))
    assert torch.equal(batch.correct_actions, expected)
    reversed_batch = generate_lifetimes(
        16, 6, seed=23, task="contextual_override", support_trials=2,
        reverse_rules=True)
    assert torch.equal(batch.frames, reversed_batch.frames)
    assert torch.equal(
        batch.correct_actions, 1 - reversed_batch.correct_actions)


def test_contextual_composition_reuses_both_acquired_primitives() -> None:
    batch = generate_lifetimes(
        16, 6, seed=27, task="contextual_composition", support_trials=2)
    assert batch.context_ids is not None
    expected = (
        batch.stimulus_identities
        ^ batch.rule_bits.unsqueeze(1)
        ^ batch.context_ids)
    assert torch.equal(batch.correct_actions, expected)

    reversed_batch = generate_lifetimes(
        16, 6, seed=27, task="contextual_composition", support_trials=2,
        reverse_rules=True)
    assert torch.equal(batch.frames, reversed_batch.frames)
    assert torch.equal(batch.context_ids, reversed_batch.context_ids)
    assert torch.equal(
        batch.correct_actions, 1 - reversed_batch.correct_actions)


def test_visible_context_counterfactual_changes_only_the_public_cue() -> None:
    batch = generate_lifetimes(16, 6, seed=25, task="visible_context")
    reversed_batch = generate_lifetimes(
        16, 6, seed=25, task="visible_context", reverse_contexts=True)
    assert batch.context_ids is not None
    assert reversed_batch.context_ids is not None
    assert not torch.equal(batch.frames, reversed_batch.frames)
    assert torch.equal(batch.context_ids, 1 - reversed_batch.context_ids)
    assert torch.equal(batch.correct_actions, batch.context_ids)
    assert torch.equal(reversed_batch.correct_actions, reversed_batch.context_ids)
    assert torch.equal(batch.correct_actions, 1 - reversed_batch.correct_actions)


def test_visible_context_xor_is_a_pixel_level_composition_atom() -> None:
    batch = generate_lifetimes(
        16, 6, seed=29, task="visible_context_xor")
    reversed_batch = generate_lifetimes(
        16, 6, seed=29, task="visible_context_xor",
        reverse_contexts=True)
    assert batch.context_ids is not None
    assert reversed_batch.context_ids is not None
    assert torch.equal(
        batch.correct_actions,
        batch.stimulus_identities ^ batch.context_ids)
    assert torch.equal(batch.stimulus_identities,
                       reversed_batch.stimulus_identities)
    assert torch.equal(batch.context_ids, 1 - reversed_batch.context_ids)
    assert torch.equal(
        batch.correct_actions, 1 - reversed_batch.correct_actions)
    plain = generate_lifetimes(
        16, 6, seed=29, task="visible_context")
    center = batch.frames.shape[-1] // 2
    assert torch.all(
        batch.frames[:, :, :, 2:5, center - 2:center + 3] == 0.98)
    assert not torch.equal(batch.frames, plain.frames)


def test_every_contextual_operation_is_observationally_distinct() -> None:
    """No two requested operations may share a rendering.

    Identical frames with conflicting correct actions make the later task
    unlearnable in principle and silently cap accuracy at chance, so the
    operation cue must separate every contextual task we train on.
    """
    tasks = (
        "visible_context", "visible_context_xor", "contextual_composition",
        "contextual_override")
    batches = {
        task: generate_lifetimes(16, 6, seed=31, task=task, support_trials=1)
        for task in tasks
    }
    for task, batch in batches.items():
        assert batch.context_ids is not None, task
    for first in tasks:
        for second in tasks:
            if first >= second:
                continue
            # Public events are shared by construction at a fixed seed, so any
            # separation has to come from the cue rather than the content.
            assert torch.equal(
                batches[first].stimulus_identities,
                batches[second].stimulus_identities)
            assert torch.equal(
                batches[first].context_ids, batches[second].context_ids)
            assert not torch.equal(
                batches[first].frames, batches[second].frames), (
                    f"{first} and {second} render identically")


def test_operation_cues_never_occlude_stimulus_or_context_pixels() -> None:
    """A cue announces the operation; it must not damage the content.

    A cue bar overlapping a glyph silently removes identity evidence, which
    looks like a hard task rather than a broken rendering.
    """
    from experiments.unified_cognitive_controller.environment import (
        _MASK_BANKS, _OPERATION_CUE_SLOTS, IMAGE_SIZE)
    glyphs = torch.zeros(IMAGE_SIZE, IMAGE_SIZE, dtype=torch.bool)
    for bank in _MASK_BANKS.values():
        glyphs |= bank.sum(dim=(0, 1)) > 0
    contexts = torch.zeros_like(glyphs)
    for y, x in ((3, 3), (IMAGE_SIZE - 4, IMAGE_SIZE - 4)):
        contexts[y - 1:y + 2, x - 1:x + 2] = True
    assert _OPERATION_CUE_SLOTS, "expected at least one cued operation"
    masks = {}
    for task, slot in _OPERATION_CUE_SLOTS.items():
        (first, last), (left, right) = slot[:2]
        cue = torch.zeros_like(glyphs)
        cue[first:last, left:right] = True
        assert cue.any(), task
        assert not (cue & glyphs).any(), f"{task} cue overlaps a stimulus glyph"
        assert not (cue & contexts).any(), f"{task} cue overlaps a context bit"
        masks[task] = cue
    # Two operations sharing a slot would render identically, which is the
    # aliasing this whole cue scheme exists to prevent.
    for first_task, first_mask in masks.items():
        for second_task, second_mask in masks.items():
            if first_task >= second_task:
                continue
            assert not (first_mask & second_mask).any(), (
                f"{first_task} and {second_task} share a cue slot")


def test_contextual_composition_cue_does_not_disturb_the_xor_slot() -> None:
    """The XOR cue is fixed by consolidated controllers and must not move."""
    composition = generate_lifetimes(
        16, 6, seed=33, task="contextual_composition", support_trials=1)
    xor = generate_lifetimes(
        16, 6, seed=33, task="visible_context_xor", support_trials=1)
    plain = generate_lifetimes(
        16, 6, seed=33, task="visible_context", support_trials=1)
    from experiments.unified_cognitive_controller.environment import (
        _OPERATION_CUE_SLOTS)
    size = composition.frames.shape[-1]
    center = size // 2
    columns = slice(center - 2, center + 3)
    # The XOR span is a literal on purpose: promoted controllers read that
    # exact band, so it is a contract rather than a detail to follow around.
    assert _OPERATION_CUE_SLOTS["visible_context_xor"][:2] == ((2, 5), (14, 19))
    composition_rows = _OPERATION_CUE_SLOTS["contextual_composition"][0]
    assert composition_rows != (2, 5)
    # Each operation lights its own slot and leaves the other one alone.
    assert torch.all(xor.frames[:, :, :, 2:5, columns] == 0.98)
    assert torch.equal(
        composition.frames[:, :, :, 2:5, columns],
        plain.frames[:, :, :, 2:5, columns])
    first, last = composition_rows
    assert torch.all(composition.frames[:, :, :, first:last, columns] == 0.98)
    assert torch.equal(
        xor.frames[:, :, :, first:last, columns],
        plain.frames[:, :, :, first:last, columns])
    # Removing a cue bar recovers the direct-context rendering exactly, which
    # is what the cue-ablation audit relies on.
    for batch, rows in ((xor, (2, 5)), (composition, composition_rows)):
        restored = batch.frames.clone()
        restored[:, :, :, rows[0]:rows[1], columns] = (
            plain.frames[:, :, :, rows[0]:rows[1], columns])
        assert torch.equal(restored, plain.frames)


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


def test_zero_initialized_relation_adapter_preserves_behavior() -> None:
    base = UnifiedCognitiveController(
        width=32, workspace_slots=4, intention_width=8)
    adapted = UnifiedCognitiveController(
        width=32, workspace_slots=4, intention_width=8,
        relation_adapter_width=16, relation_adapter_gated=True,
        action_adapter_width=16,
        action_adapter_gated=True)
    missing, unexpected = adapted.load_state_dict(base.state_dict(), strict=False)
    assert not unexpected
    assert set(missing) == {
        name for name in adapted.state_dict()
        if name.startswith((
            "relation_adapter.", "relation_adapter_gate.",
            "action_adapter.", "action_adapter_gate."))}
    batch = generate_lifetimes(8, 4, seed=24)
    base_result = rollout(base, batch, sample_actions=False)
    adapted_result = rollout(adapted, batch, sample_actions=False)
    assert torch.equal(base_result["logits"], adapted_result["logits"])
    assert torch.equal(base_result["actions"], adapted_result["actions"])


def test_skill_adapter_stack_inserts_exactly_behavior_preserving_slots(
        ) -> None:
    """Successor slots must be free to add and inert until they are trained."""
    occupied = UnifiedCognitiveController(
        width=32, workspace_slots=4, intention_width=8,
        relation_adapter_width=16, relation_adapter_gated=True,
        action_adapter_width=16, action_adapter_gated=True)
    # An empty stack contributes no state at all, so promoted checkpoints
    # written before the stack existed still load strictly.
    assert not [
        name for name in occupied.state_dict()
        if name.startswith("skill_adapter")]
    for slots in ((16,), (16, 24)):
        extended = UnifiedCognitiveController(
            width=32, workspace_slots=4, intention_width=8,
            relation_adapter_width=16, relation_adapter_gated=True,
            action_adapter_width=16, action_adapter_gated=True,
            skill_adapter_widths=slots)
        missing, unexpected = extended.load_state_dict(
            occupied.state_dict(), strict=False)
        assert not unexpected
        assert set(missing) == {
            name for name in extended.state_dict()
            if name.startswith("skill_adapter")}
        assert len(extended.skill_adapters) == len(slots)
        batch = generate_lifetimes(8, 4, seed=41)
        before = rollout(occupied, batch, sample_actions=False)
        after = rollout(extended, batch, sample_actions=False)
        assert torch.equal(before["logits"], after["logits"])
        assert torch.equal(before["actions"], after["actions"])
    # A trained slot must actually be able to change behavior, otherwise the
    # exactness above would be hiding a dead module.
    live = UnifiedCognitiveController(
        width=32, workspace_slots=4, intention_width=8,
        skill_adapter_widths=(16,))
    with torch.no_grad():
        live.skill_adapters[0][-1].bias.fill_(1.5)
    plain = UnifiedCognitiveController(
        width=32, workspace_slots=4, intention_width=8)
    live.load_state_dict(plain.state_dict(), strict=False)
    with torch.no_grad():
        live.skill_adapters[0][-1].bias.fill_(1.5)
    batch = generate_lifetimes(8, 4, seed=42)
    assert not torch.equal(
        rollout(plain, batch, sample_actions=False)["logits"],
        rollout(live, batch, sample_actions=False)["logits"])


def test_rectified_slot_gate_can_shut_exactly_and_sigmoid_cannot() -> None:
    """Exact zero is the whole point: only it leaves an old skill untouched.

    A sigmoid gate is bounded away from zero, so a slot always perturbs every
    event it sees. That residual perturbation is what accumulates into the
    nearest-neighbour skill rung after rung.
    """
    features = torch.randn(16, 64)
    for mode, reachable in (("relu", True), ("sigmoid", False)):
        model = UnifiedCognitiveController(
            width=32, workspace_slots=4, intention_width=8,
            skill_adapter_widths=(16,), skill_adapter_gate_mode=mode)
        gate = model.skill_adapter_gates[0]
        with torch.no_grad():
            gate.bias.fill_(-5.0)
            gate.weight.zero_()
        opening = (
            torch.relu(gate(features)) if mode == "relu"
            else torch.sigmoid(gate(features)))
        assert (opening == 0).all().item() is reachable, mode
    # Both modes still insert exactly behavior-preserving slots.
    base = UnifiedCognitiveController(
        width=32, workspace_slots=4, intention_width=8)
    batch = generate_lifetimes(8, 4, seed=61)
    for mode in ("relu", "sigmoid"):
        extended = UnifiedCognitiveController(
            width=32, workspace_slots=4, intention_width=8,
            skill_adapter_widths=(16,), skill_adapter_gate_mode=mode)
        extended.load_state_dict(base.state_dict(), strict=False)
        assert torch.equal(
            rollout(base, batch, sample_actions=False)["logits"],
            rollout(extended, batch, sample_actions=False)["logits"]), mode


def test_slot_gate_hidden_layer_is_optional_and_preserves_behavior() -> None:
    base = UnifiedCognitiveController(
        width=32, workspace_slots=4, intention_width=8)
    batch = generate_lifetimes(8, 4, seed=62)
    for hidden in (0, 16):
        model = UnifiedCognitiveController(
            width=32, workspace_slots=4, intention_width=8,
            skill_adapter_widths=(16,), skill_adapter_gate_mode="relu",
            skill_adapter_gate_hidden=hidden)
        model.load_state_dict(base.state_dict(), strict=False)
        assert torch.equal(
            rollout(base, batch, sample_actions=False)["logits"],
            rollout(model, batch, sample_actions=False)["logits"]), hidden


def test_controller_reports_slot_openings_and_residual_norms() -> None:
    model = UnifiedCognitiveController(
        width=32, workspace_slots=4, intention_width=8,
        skill_adapter_widths=(16, 16))
    plain = UnifiedCognitiveController(
        width=32, workspace_slots=4, intention_width=8)
    assert plain.initial_state(2, device="cpu") is not None
    batch = generate_lifetimes(4, 4, seed=63)
    state = model.initial_state(4, device="cpu")
    output, _ = model.step(
        batch.frames[:, 0], state,
        torch.zeros(4, dtype=torch.long), torch.zeros(4), torch.zeros(4))
    assert output.skill_adapter_openings is not None
    assert output.skill_adapter_openings.shape == (4, 2)
    assert output.skill_adapter_residual_norms is not None
    assert output.skill_adapter_residual_norms.shape == (4, 2)
    # A freshly inserted slot outputs exactly zero, so it perturbs nothing.
    assert torch.equal(
        output.skill_adapter_residual_norms,
        torch.zeros_like(output.skill_adapter_residual_norms))
    # A controller with no successor slots reports nothing rather than zeros.
    plain_output, _ = plain.step(
        batch.frames[:, 0], plain.initial_state(4, device="cpu"),
        torch.zeros(4, dtype=torch.long), torch.zeros(4), torch.zeros(4))
    assert plain_output.skill_adapter_openings is None
    assert plain_output.skill_adapter_residual_norms is None


def test_skill_adapter_gate_mode_is_validated() -> None:
    try:
        UnifiedCognitiveController(
            width=32, workspace_slots=4, intention_width=8,
            skill_adapter_widths=(16,), skill_adapter_gate_mode="tanh")
    except ValueError:
        return
    raise AssertionError("accepted an unknown skill adapter gate mode")


def test_skill_adapter_widths_are_validated() -> None:
    for slots in ((0,), (16, -1)):
        try:
            UnifiedCognitiveController(
                width=32, workspace_slots=4, intention_width=8,
                skill_adapter_widths=slots)
        except ValueError:
            continue
        raise AssertionError(f"accepted invalid skill adapter widths {slots}")


def test_attempted_loss_has_no_unattempted_target_argument() -> None:
    logits = torch.tensor([[0.2, -0.4], [0.3, 0.8]], requires_grad=True)
    actions = torch.tensor([0, 1])
    outcomes = torch.tensor([1.0, 0.0])
    loss = attempted_success_loss(logits, actions, outcomes)
    loss.backward()
    assert logits.grad is not None
    assert logits.grad[0, 1] == 0
    assert logits.grad[1, 0] == 0


def test_passive_critic_features_describe_only_attempted_option() -> None:
    features = torch.arange(3 * 4 * 8, dtype=torch.float32).reshape(3, 4, 8)
    scores = torch.tensor([
        [0.0, 1.0, 2.0, 3.0],
        [2.0, 1.0, 0.0, -1.0],
        [0.1, 0.2, 0.3, 0.4],
    ])
    actions = torch.tensor([1, 0, 3])
    propensities = torch.tensor([0.2, 0.4, 0.6])
    critic_features = attempted_action_features(
        features, scores, actions, propensities)
    assert critic_features.shape == (3, CRITIC_INPUT_WIDTH)
    selected = features[torch.arange(3), actions]
    assert torch.equal(critic_features[:, 18:26], selected)
    assert torch.equal(critic_features[:, 26], propensities)


def test_passive_critic_controls_remove_only_registered_evidence() -> None:
    features = torch.randn(5, CRITIC_INPUT_WIDTH)
    missing_action = apply_evidence_control(features, "missing_action")
    missing_context = apply_evidence_control(features, "missing_context")
    assert torch.equal(missing_action[:, :18], features[:, :18])
    assert torch.count_nonzero(missing_action[:, 18:]) == 0
    assert torch.count_nonzero(missing_context[:, :18]) == 0
    assert torch.equal(missing_context[:, 18:], features[:, 18:])
    action_only = critic_evidence(
        features, "intact", primary_evidence="action_only")
    action_only_missing = critic_evidence(
        features, "missing_action", primary_evidence="action_only")
    assert torch.count_nonzero(action_only[:, :18]) == 0
    assert torch.equal(action_only[:, 18:], features[:, 18:])
    assert torch.count_nonzero(action_only_missing) == 0
    action_query = critic_evidence(
        features, "intact", primary_evidence="action_query")
    query_missing = critic_evidence(
        features, "missing_context", primary_evidence="action_query")
    action_missing = critic_evidence(
        features, "missing_action", primary_evidence="action_query")
    assert torch.equal(action_query[:, :4], features[:, :4])
    assert torch.count_nonzero(action_query[:, 4:18]) == 0
    assert torch.count_nonzero(query_missing[:, :18]) == 0
    assert torch.equal(query_missing[:, 18:], features[:, 18:])
    assert torch.equal(action_missing[:, :4], features[:, :4])
    assert torch.count_nonzero(action_missing[:, 18:]) == 0
    query_only = critic_evidence(
        features, "intact", primary_evidence="query_only")
    query_only_missing = critic_evidence(
        features, "missing_action", primary_evidence="query_only")
    assert torch.equal(query_only[:, :4], features[:, :4])
    assert torch.count_nonzero(query_only[:, 4:]) == 0
    assert torch.count_nonzero(query_only_missing) == 0


def test_passive_critic_logging_propensities_are_exact_mixture() -> None:
    scores = torch.tensor([[0.0, 1.0, 2.0]])
    probabilities = exploration_probabilities(
        scores, epsilon=0.3, temperature=2.0)
    expected = 0.7 * torch.softmax(scores / 2.0, dim=-1) + 0.1
    assert torch.allclose(probabilities, expected)
    assert torch.allclose(probabilities.sum(-1), torch.ones(1))


def test_passive_critic_concordance_handles_order_and_ties() -> None:
    outcomes = torch.tensor([0.0, 0.5, 1.0])
    assert concordance(outcomes, outcomes) == pytest.approx(1.0)
    assert concordance(-outcomes, outcomes) == pytest.approx(0.0)
    assert concordance(torch.zeros_like(outcomes), outcomes) == pytest.approx(
        0.5)


def test_shadow_compute_loss_uses_only_attempted_action_outcome() -> None:
    logits = torch.tensor(
        [[0.1, 0.2], [0.3, 0.4]], requires_grad=True)
    loss = selected_compute_loss(
        logits, torch.tensor([0, 1]), torch.tensor([1.0, 0.0]))
    loss.backward()
    assert logits.grad is not None
    assert logits.grad[0, 1] == 0
    assert logits.grad[1, 0] == 0


def test_shadow_compute_controls_remove_or_mismatch_only_evidence() -> None:
    features = torch.arange(20, dtype=torch.float32).reshape(5, 4)
    permutation = torch.tensor([4, 3, 2, 1, 0])
    assert torch.equal(
        controlled_features(features, "intact"), features)
    assert torch.equal(
        controlled_features(
            features, "feature_shuffled", permutation=permutation),
        features[permutation])
    assert torch.count_nonzero(
        controlled_features(features, "missing_evidence")) == 0


def test_shadow_compute_audit_rewards_context_sensitive_choice() -> None:
    critic = ShadowComputeCritic(hidden=4)
    with torch.no_grad():
        # Make the read action rise with feature zero and the no-read action
        # fall with it, yielding a known context-sensitive decision.
        critic.network[1].weight.zero_()
        critic.network[1].bias.zero_()
        critic.network[-1].weight.zero_()
        critic.network[-1].bias.copy_(torch.tensor([1.0, -1.0]))
    features = torch.zeros(4, 4)
    no_read = torch.ones(4)
    read = torch.zeros(4)
    metrics = _shadow_metrics(
        critic, features, no_read, read, torch.zeros(2),
        read_cost=0.01)
    assert metrics["read_rate"] == 0.0
    assert metrics["compute_choice_accuracy"] == 1.0
    assert metrics["shadow_verified_utility"] == 1.0


def test_attempted_advantage_target_is_unbiased_under_uniform_logging() -> None:
    # For one context, no-read utility=.2 and read utility=.8. Averaging the
    # two logged pseudo-targets must recover the true advantage .6.
    actions = torch.tensor([0, 1])
    utilities = torch.tensor([0.2, 0.8])
    targets = attempted_advantage_target(
        actions, utilities, baseline=0.5, propensity=0.5)
    assert targets.mean() == pytest.approx(0.6)
    # An action-independent baseline cancels from the expectation.
    shifted = attempted_advantage_target(
        actions, utilities, baseline=0.1, propensity=0.5)
    assert shifted.mean() == pytest.approx(0.6)


def test_thought_transfer_controls_only_remove_or_mismatch_evidence() -> None:
    features = torch.arange(20, dtype=torch.float32).reshape(5, 4)
    permutation = torch.tensor([4, 3, 2, 1, 0])
    assert torch.equal(
        active_thought_features(features, "inherited"), features)
    assert torch.equal(
        active_thought_features(
            features, "feature_shuffled", permutation),
        features[permutation])
    assert torch.count_nonzero(
        active_thought_features(features, "missing_evidence")) == 0


def test_thought_transfer_metrics_reward_context_sensitive_choice() -> None:
    from .train_shadow_compute_advantage import ComputeAdvantageHead

    head = ComputeAdvantageHead(hidden=2)
    with torch.no_grad():
        head.network[1].weight.zero_()
        head.network[1].bias.zero_()
        head.network[-1].weight.zero_()
        head.network[-1].bias.fill_(1.0)
    features = torch.zeros(4, 4)
    immediate = torch.zeros(4)
    thought = torch.ones(4)
    metrics = thought_compute_metrics(
        head, features, immediate, thought, thought_cost=0.01)
    assert metrics["thought_rate"] == 1.0
    assert metrics["compute_choice_accuracy"] == 1.0
    assert metrics["verified_utility"] == pytest.approx(0.99)


def test_thought_transfer_stable_bits_requires_persistent_crossing() -> None:
    def row(bits: int, choice: float, gain: float, gap: float) -> dict:
        return {
            "unique_verifier_bits": bits,
            "compute_choice_accuracy": choice,
            "verified_utility": 0.5 + gain,
            "strongest_fixed_utility": 0.5,
            "captured_oracle_gap_fraction": gap,
        }

    history = [
        row(0, 0.50, 0.00, 0.00),
        row(120, 0.70, 0.12, 0.30),
        row(240, 0.60, 0.11, 0.25),
        row(360, 0.68, 0.13, 0.35),
        row(480, 0.72, 0.14, 0.40),
    ]
    assert stable_thought_bits(history) == 360


def test_shared_compute_value_copies_source_but_resets_novel_adapter() -> None:
    from .train_shadow_compute_advantage import ComputeAdvantageHead

    source = ComputeAdvantageHead(hidden=4)
    with torch.no_grad():
        for parameter in source.parameters():
            parameter.fill_(0.25)
    shared = SharedComputeValue(hidden=4)
    initialize_from_advantage(shared, {
        "head_state_dict": source.state_dict(),
    })
    features = torch.randn(7, 4)
    assert torch.equal(
        source(features), shared(features, "read"))
    assert torch.count_nonzero(shared(features, "thought")) == 0


def test_cost_aware_expansion_preserves_old_read_at_zero_cost() -> None:
    from .train_shadow_compute_advantage import ComputeAdvantageHead

    source = ComputeAdvantageHead(hidden=4)
    expanded = CostAwareComputeValue(hidden=4)
    initialize_from_four_feature(expanded, {
        "head_state_dict": source.state_dict(),
    })
    features = torch.randn(7, 4)
    costs = torch.zeros(7)
    assert torch.equal(
        source(features), expanded(features, costs, "read"))
    assert torch.count_nonzero(
        expanded(features, costs, "requery")) == 0


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


def test_disk_latent_memory_compacts_selected_history_exactly(
        tmp_path: Path) -> None:
    memory = DiskLatentMemory(width=4, capacity=4)
    keys = torch.eye(4)
    values = torch.arange(16, dtype=torch.float32).reshape(4, 4)
    memory.commit(keys, values, torch.ones(4), threshold=0.0)
    memory.store.access_count.copy_(torch.tensor([1, 2, 3, 4]))
    memory.store.success_count.copy_(torch.tensor([4, 3, 2, 1]))
    memory.store.volatility.copy_(torch.tensor([0.1, 0.2, 0.8, 0.9]))
    compact = memory.compact([0, 2])
    assert compact.count == 2
    assert compact.store.capacity == 2
    assert torch.equal(compact.store.keys, keys[[0, 2]])
    assert compact.store.access_count.tolist() == [1, 3]
    assert compact.store.success_count.tolist() == [4, 2]
    assert torch.allclose(
        compact.store.volatility, torch.tensor([0.1, 0.8]))
    path = tmp_path / "compact.pt"
    compact.save(path)
    restored = DiskLatentMemory.load(path)
    assert restored.store.capacity == 2
    assert torch.equal(restored.store.keys, compact.store.keys)
    assert torch.equal(
        restored.store.volatility, compact.store.volatility)


def test_bankwise_physical_survival_uses_only_protection_scalar() -> None:
    from .audit_adaptive_physical_pruning import (
        bankwise_survival_mask,
        shuffled_bankwise_mask,
    )

    bank = {
        "valid": torch.ones(4, 6, dtype=torch.bool),
        "representative_ranks": torch.tensor(
            [[0, 0, 1, 1, 2, 2]]).repeat(4, 1),
    }
    scores = torch.zeros(4, 6)
    scores[1, 4] = 1.0
    scores[3, 5] = 2.0
    selected = bankwise_survival_mask(
        bank, scores, threshold=1.0)
    assert selected.sum(-1).tolist() == [4, 6, 4, 6]
    shuffled = shuffled_bankwise_mask(
        bank, selected, seed=19)
    assert shuffled.sum(-1).sort().values.tolist() == [4, 4, 6, 6]
    assert not torch.equal(selected, shuffled)


def test_tiered_memory_promotes_thaws_and_preserves_cold_archive(
        tmp_path: Path) -> None:
    cold = DiskLatentMemory(width=6, capacity=6)
    keys = torch.eye(6)
    values = torch.arange(36, dtype=torch.float32).reshape(6, 6)
    cold.commit(keys, values, torch.ones(6), threshold=0.0)
    ranks = torch.tensor([0, 0, 1, 1, 2, 2])
    memory = TieredLatentMemory(
        cold, ranks, protection=0.0, threshold=0.5)
    assert memory.hot().count == 4
    memory.observe_verified_rescue(True, decay=0.9)
    assert memory.hot().count == 6
    for _ in range(7):
        memory.observe_verified_rescue(False, decay=0.9)
    assert memory.protection < 0.5
    assert memory.hot().count == 4
    assert memory.cold.count == 6

    path = tmp_path / "tiered"
    memory.save(path)
    restored = TieredLatentMemory.load(path)
    assert restored.protection == pytest.approx(memory.protection)
    assert restored.threshold == memory.threshold
    assert torch.equal(restored.representative_ranks, ranks)
    assert torch.equal(restored.cold.store.keys, keys)
    assert restored.hot().count == 4


def test_hot_memory_decay_selection_is_accuracy_constrained() -> None:
    from .select_hot_memory_decay import select_candidate

    def report(
            decay: float, accepted: bool, rows: float,
            accuracy: float) -> dict[str, object]:
        return {
            "decay": decay,
            "gates": {"accepted": accepted},
            "phases": {
                "easy_interlude": {
                    "decaying": {"mean_hot_rows": rows}},
                "hard_return": {
                    "decaying": {
                        "first_attempt_accuracy": accuracy}},
            },
            "reactivation": {
                "last_four_gain_over_fixed_core": 0.01,
                "last_four_gain_over_shuffled_evidence": 0.01,
            },
        }

    selected, _ = select_candidate([
        report(0.5, False, 4.0, 0.90),
        report(0.9, True, 4.1, 0.99),
        report(0.95, True, 4.2, 0.995),
    ])
    assert selected["decay"] == 0.9


def test_usage_prior_scale_can_restore_exact_content_retrieval() -> None:
    memory = DiskLatentMemory(width=2, capacity=2)
    keys = torch.tensor([[1.0, 0.0], [0.8, 0.6]])
    values = torch.eye(2)
    memory.commit(
        keys, values, torch.tensor([0.1, 1.0]), threshold=0.0)
    query = keys[:1]
    prior_read, _ = memory.retrieve(
        query, top_k=1, confidence_mode="cosine",
        usage_prior_scale=1.0)
    content_read, _ = memory.retrieve(
        query, top_k=1, confidence_mode="cosine",
        usage_prior_scale=0.0)
    assert torch.equal(prior_read, values[1:2])
    assert torch.equal(content_read, values[:1])
    memory.store.record_outcomes(
        query, torch.ones(1), usage_prior_scale=0.0)
    assert memory.store.success_count.tolist() == [1, 0]


def test_read_receipt_preserves_causal_row_under_unequal_strengths() -> None:
    memory = DiskLatentMemory(width=2, capacity=2)
    keys = torch.tensor([[1.0, 0.0], [0.8, 0.6]])
    values = torch.eye(2)
    memory.commit(
        keys, values, torch.tensor([0.1, 1.0]), threshold=0.0)
    query = keys[:1]
    # The ordinary strength-prior read is intentionally redirected to row 1.
    ordinary, _ = memory.retrieve(
        query, top_k=1, confidence_mode="cosine", usage_prior_scale=1.0)
    assert torch.equal(ordinary, values[1:2])
    recalled, _, receipt = memory.retrieve_with_receipt(
        query, top_k=1, confidence_mode="cosine", usage_prior_scale=0.0)
    assert torch.equal(recalled, values[:1])
    assert receipt.tolist() == [0]
    memory.record_outcomes_from_receipts(
        receipt, torch.ones(1), update_volatility=True,
        stale_thaw_rate=0.0)
    assert memory.store.success_count.tolist() == [1, 0]


def test_receipt_outcomes_keep_repeated_rows_aligned() -> None:
    memory = DiskLatentMemory(width=4, capacity=2)
    keys = torch.eye(4)[:2]
    memory.commit(keys, keys, torch.ones(2), threshold=0.0)
    _, _, receipts = memory.retrieve_with_receipt(
        keys[:1].repeat(3, 1), top_k=1, confidence_mode="cosine",
        usage_prior_scale=0.0)
    memory.record_outcomes_from_receipts(
        receipts, torch.tensor([1.0, 0.0, 1.0]))
    assert memory.store.success_count.tolist() == [2, 0]
    assert memory.store.failure_count.tolist() == [1, 0]


def test_transaction_rejects_candidate_that_forgets_old_skill() -> None:
    memory = DiskLatentMemory(width=4, capacity=2)
    old_keys = torch.eye(4)[:2]
    old_values = old_keys.clone()
    memory.commit(old_keys, old_values, torch.ones(2), threshold=0.0)
    new_key = torch.tensor([0.0, 0.0, 1.0, 0.0])
    new_value = new_key.clone()

    def old_skill(store: DiskLatentMemory) -> float:
        read, _ = store.retrieve(
            old_keys[:1], top_k=1, confidence_mode="cosine",
            usage_prior_scale=0.0)
        return float(torch.nn.functional.cosine_similarity(
            read, old_values[:1]).item())

    def new_skill(store: DiskLatentMemory) -> float:
        read, _ = store.retrieve(
            new_key.unsqueeze(0), top_k=1, confidence_mode="cosine",
            usage_prior_scale=0.0)
        return float(torch.nn.functional.cosine_similarity(
            read, new_value.unsqueeze(0)).item())

    result = memory.transactional_replace(
        0, new_key, new_value, 1.0, [old_skill], new_skill,
        required_candidate_gain=0.5, rejection_penalty=0.25)
    assert not result.committed
    assert result.maximum_retention_drop > 0.5
    assert torch.equal(result.memory.store.keys, memory.store.keys)
    assert torch.equal(result.memory.store.values, memory.store.values)


def test_transaction_commits_safe_candidate_and_preserves_disk_state(
        tmp_path: Path) -> None:
    memory = DiskLatentMemory(width=4, capacity=2)
    old_keys = torch.eye(4)[:2]
    memory.commit(old_keys, old_keys, torch.ones(2), threshold=0.0)
    new_key = torch.tensor([0.0, 0.0, 1.0, 0.0])

    def old_skill(store: DiskLatentMemory) -> float:
        read, _ = store.retrieve(
            old_keys[:1], top_k=1, confidence_mode="cosine",
            usage_prior_scale=0.0)
        return float(torch.nn.functional.cosine_similarity(
            read, old_keys[:1]).item())

    def new_skill(store: DiskLatentMemory) -> float:
        read, _ = store.retrieve(
            new_key.unsqueeze(0), top_k=1, confidence_mode="cosine",
            usage_prior_scale=0.0)
        return float(torch.nn.functional.cosine_similarity(
            read, new_key.unsqueeze(0)).item())

    result = memory.transactional_replace(
        1, new_key, new_key, 1.0, [old_skill], new_skill,
        required_candidate_gain=0.5)
    assert result.committed
    assert result.maximum_retention_drop == 0.0
    assert result.candidate_gain >= 0.5
    path = tmp_path / "transaction-committed.pt"
    result.memory.save(path)
    restored = DiskLatentMemory.load(path)
    assert torch.equal(restored.store.keys, result.memory.store.keys)
    assert torch.equal(restored.store.values, result.memory.store.values)


def test_usage_prior_scale_can_vary_per_query() -> None:
    memory = DiskLatentMemory(width=2, capacity=2)
    keys = torch.tensor([[1.0, 0.0], [0.8, 0.6]])
    values = torch.eye(2)
    memory.commit(
        keys, values, torch.tensor([0.1, 1.0]), threshold=0.0)
    reads, _ = memory.retrieve(
        keys[:1].repeat(2, 1), top_k=1,
        confidence_mode="cosine",
        usage_prior_scale=torch.tensor([0.0, 1.0]))
    assert torch.equal(reads, values)


def test_disk_memory_records_persists_and_resets_access_counts(
        tmp_path: Path) -> None:
    memory = DiskLatentMemory(width=4, capacity=2)
    keys = torch.eye(4)[:2]
    values = torch.arange(8, dtype=torch.float32).reshape(2, 4)
    assert memory.commit(
        keys, values, torch.ones(2), threshold=0.0) == 2
    memory.retrieve(
        keys[:1].repeat(3, 1), top_k=1,
        confidence_mode="cosine", record_access=True)
    memory.store.record_outcomes(
        keys[:1].repeat(3, 1), torch.tensor([1.0, 1.0, 0.0]))
    assert memory.store.access_count.tolist() == [3, 0]
    assert memory.store.success_count.tolist() == [2, 0]
    assert memory.store.failure_count.tolist() == [1, 0]
    path = tmp_path / "access-counts.pt"
    memory.save(path)
    restored = DiskLatentMemory.load(path)
    assert restored.store.access_count.tolist() == [3, 0]
    assert restored.store.success_count.tolist() == [2, 0]
    assert restored.store.failure_count.tolist() == [1, 0]
    restored.replace(0, keys[1], values[1], 0.9)
    assert restored.store.access_count.tolist() == [0, 0]
    assert restored.store.success_count.tolist() == [0, 0]
    assert restored.store.failure_count.tolist() == [0, 0]


def test_verified_outcomes_protect_successful_rows_and_thaw_failures() -> None:
    memory = DiskLatentMemory(width=4, capacity=3)
    keys = torch.eye(4)[:3]
    memory.commit(keys, keys, torch.ones(3), threshold=0.0)
    memory.store.record_outcomes(
        keys, torch.tensor([1.0, 0.0, 1.0]),
        update_volatility=True, success_protection_rate=0.5,
        failure_thaw_rate=0.5, stale_thaw_rate=0.0)
    assert torch.allclose(
        memory.store.volatility[:3],
        torch.tensor([0.5, 1.0, 0.5]))
    memory.store.record_outcomes(
        keys[1:2], torch.tensor([1.0]),
        update_volatility=True, success_protection_rate=0.5,
        failure_thaw_rate=0.5, stale_thaw_rate=0.1)
    assert torch.allclose(
        memory.store.volatility[:3],
        torch.tensor([0.55, 0.5, 0.55]))


def test_volatility_persists_and_old_memories_default_to_plastic(
        tmp_path: Path) -> None:
    memory = DiskLatentMemory(width=4, capacity=2)
    memory.commit(
        torch.eye(4)[:1], torch.eye(4)[:1],
        torch.ones(1), threshold=0.0)
    memory.store.volatility[0] = 0.125
    path = tmp_path / "volatility.pt"
    memory.save(path)
    restored = DiskLatentMemory.load(path)
    assert restored.store.volatility.tolist() == [0.125, 1.0]


def test_elastic_replace_obeys_row_volatility() -> None:
    memory = DiskLatentMemory(width=2, capacity=2)
    keys = torch.eye(2)
    memory.commit(keys, keys, torch.ones(2), threshold=0.0)
    memory.store.volatility[:2] = torch.tensor([0.0, 1.0])
    frozen_before = memory.store.values[0].clone()
    assert memory.elastic_replace(0, keys[1], -keys[1], 0.5) == 0.0
    assert torch.equal(memory.store.values[0], frozen_before)
    assert memory.elastic_replace(1, keys[0], -keys[0], 0.5) == 1.0
    assert torch.equal(memory.store.values[1], -keys[0])


def test_outcome_order_volatility_preserves_temporal_information() -> None:
    from .train_controller_memory_volatility import outcome_order_volatility

    failures_then_successes = torch.tensor(
        [[0.0] * 5 + [1.0] * 5])
    successes_then_failures = 1.0 - failures_then_successes
    stable = outcome_order_volatility(failures_then_successes)
    decoy = outcome_order_volatility(successes_then_failures)
    assert failures_then_successes.sum() == successes_then_failures.sum()
    assert stable.item() < decoy.item() - 0.4


def test_volatility_expansion_is_exactly_behavior_preserving() -> None:
    from .train_controller_memory_volatility import expand_with_volatility

    source = UnifiedCognitiveController(
        width=16, workspace_slots=2, intention_width=4,
        adaptive_memory_replace=True,
        adaptive_memory_replace_hidden=4,
        adaptive_memory_replace_features=7)
    payload = {
        "model_configuration": {
            "width": 16,
            "workspace_slots": 2,
            "intention_width": 4,
            "adaptive_memory_replace": True,
            "adaptive_memory_replace_hidden": 4,
            "adaptive_memory_replace_features": 7,
        },
        "state_dict": source.state_dict(),
    }
    expanded, configuration = expand_with_volatility(
        payload, device=torch.device("cpu"))
    old_features = torch.randn(11, 5, 7)
    new_features = torch.cat((
        old_features, torch.randn(11, 5, 1)), dim=-1)
    assert configuration["adaptive_memory_replace_features"] == 8
    assert torch.equal(
        source.memory_replacement_scores(old_features),
        expanded.memory_replacement_scores(new_features))
    assert expanded.memory_replacement_extra_gate.weight[0, 2] == 0


def test_usage_prior_expansion_preserves_controller_and_starts_at_one() -> None:
    from .train_memory_usage_prior_race import (
        expand_with_adaptive_usage_prior,
    )

    source = UnifiedCognitiveController(
        width=16, workspace_slots=2, intention_width=4,
        adaptive_memory_replace=True,
        adaptive_memory_replace_hidden=4,
        adaptive_memory_replace_features=8)
    payload = {
        "model_configuration": {
            "width": 16,
            "workspace_slots": 2,
            "intention_width": 4,
            "adaptive_memory_replace": True,
            "adaptive_memory_replace_hidden": 4,
            "adaptive_memory_replace_features": 8,
        },
        "state_dict": source.state_dict(),
    }
    expanded, configuration = expand_with_adaptive_usage_prior(
        payload, device=torch.device("cpu"))
    features = torch.randn(9, 7, 8)
    assert configuration["adaptive_memory_usage_prior"] is True
    assert torch.equal(
        source.memory_replacement_scores(features),
        expanded.memory_replacement_scores(features))
    assert expanded.effective_memory_usage_prior_scale() == 1.0


def test_conditional_usage_prior_starts_with_hard_content_first_action() -> None:
    from .train_conditional_memory_usage_prior import (
        expand_with_conditional_usage_prior,
    )

    source = UnifiedCognitiveController(
        width=16, workspace_slots=2, intention_width=4,
        adaptive_memory_replace=True,
        adaptive_memory_replace_hidden=4,
        adaptive_memory_replace_features=8,
        adaptive_memory_usage_prior=True)
    with torch.no_grad():
        source.memory_usage_prior_scale.zero_()
    payload = {
        "model_configuration": {
            "width": 16,
            "workspace_slots": 2,
            "intention_width": 4,
            "adaptive_memory_replace": True,
            "adaptive_memory_replace_hidden": 4,
            "adaptive_memory_replace_features": 8,
            "adaptive_memory_usage_prior": True,
        },
        "state_dict": source.state_dict(),
    }
    expanded, configuration = expand_with_conditional_usage_prior(
        payload, hidden=8, device=torch.device("cpu"))
    features = torch.randn(32, 4)
    probability = expanded.memory_usage_prior_probability(features)
    assert configuration["adaptive_memory_usage_prior_hidden"] == 8
    assert bool((probability < 0.5).all())
    assert expanded.effective_memory_usage_prior_scale() == 0.0


def test_continuous_usage_batch_has_valid_query_dependent_intervals() -> None:
    from .train_continuous_memory_usage_prior import (
        continuous_batch,
        select_rows,
    )

    payload = torch.load(
        "artifacts/checkpoints/"
        "unified_conditional_memory_usage_prior_seed17603.pt",
        map_location="cpu", weights_only=False)
    model = UnifiedCognitiveController(**payload["model_configuration"])
    model.load_state_dict(payload["state_dict"])
    data = continuous_batch(
        model, count=32, rows=4, seed=17700,
        device=torch.device("cpu"), heldout=True)
    exact = data["arm"] == 0
    ambiguous = ~exact
    scales = torch.zeros(32)
    scales[ambiguous] = data["decision_boundary"][ambiguous] + 0.01
    selected, _ = select_rows(data, scales)
    assert torch.equal(selected, data["target_index"])
    assert bool(
        (data["decision_boundary"][ambiguous] > 0.0).all())
    assert bool(
        (data["decision_boundary"][ambiguous] < 1.0).all())


def test_four_target_memory_rows_are_all_selectable_after_permutation() -> None:
    from .train_four_target_memory_retrieval import (
        behavioral_row_outcomes,
        four_target_batch,
        select_rows,
    )

    payload = torch.load(
        "artifacts/checkpoints/"
        "unified_continuous_memory_usage_prior_seed17718.pt",
        map_location="cpu", weights_only=False)
    model = UnifiedCognitiveController(**payload["model_configuration"])
    model.load_state_dict(payload["state_dict"])
    data = four_target_batch(
        model, count=64, seed=17800,
        device=torch.device("cpu"), heldout=True)
    representative_scales = torch.tensor([0.10, 0.35, 0.50, 0.80])
    scales = representative_scales[data["target_class"]]
    selected, values = select_rows(data, scales)
    assert torch.equal(selected, data["target_slot"])
    expected = torch.gather(
        data["values"], 1,
        data["target_slot"][:, None, None].expand(
            -1, 1, model.width)).squeeze(1)
    assert torch.equal(values, expected)
    assert data["target_slot"].unique().numel() == 4
    outcomes = behavioral_row_outcomes(
        model, data, device=torch.device("cpu"))
    assert torch.equal(
        outcomes.sum(-1), torch.ones(data["target_slot"].shape[0]))
    assert torch.equal(
        outcomes.gather(1, data["target_slot"].unsqueeze(-1)).squeeze(-1),
        torch.ones(data["target_slot"].shape[0]))
    shifted = four_target_batch(
        model, count=64, seed=17801,
        device=torch.device("cpu"), heldout=True,
        boundary_shift_range=(0.04, 0.04))
    shifted_scales = (
        representative_scales[shifted["target_class"]]
        + shifted["boundary_shift"])
    shifted_selected, _ = select_rows(shifted, shifted_scales)
    assert torch.equal(shifted_selected, shifted["target_slot"])
    deformed = four_target_batch(
        model, count=64, seed=17802,
        device=torch.device("cpu"), heldout=True,
        crossing_jitter_range=(-0.02, 0.02),
        slope_jitter_range=(-0.06, 0.06))
    target_class = deformed["target_class"]
    crossings = deformed["crossings"]
    lower = torch.zeros(64)
    upper = torch.ones(64)
    lower = torch.where(target_class == 1, crossings[:, 0], lower)
    lower = torch.where(target_class == 2, crossings[:, 1], lower)
    lower = torch.where(target_class == 3, crossings[:, 2], lower)
    upper = torch.where(target_class == 0, crossings[:, 0], upper)
    upper = torch.where(target_class == 1, crossings[:, 1], upper)
    upper = torch.where(target_class == 2, crossings[:, 2], upper)
    deformed_selected, _ = select_rows(
        deformed, (lower + upper) / 2)
    assert torch.equal(deformed_selected, deformed["target_slot"])


def test_verified_scale_interval_penalizes_only_behavior_changes() -> None:
    from .train_four_target_memory_retrieval import scale_interval_loss

    predicted = torch.tensor([0.20, 0.50, 0.90], requires_grad=True)
    candidates = torch.tensor([0.0, 0.5, 1.0])
    allowed = torch.tensor([
        [True, True, False],
        [False, True, False],
        [True, True, False],
    ])
    loss = scale_interval_loss(predicted, allowed, candidates)
    assert loss.item() == pytest.approx((0.90 - 0.50) ** 2 / 3)
    loss.backward()
    assert predicted.grad is not None
    assert torch.equal(
        predicted.grad[:2], torch.zeros_like(predicted.grad[:2]))
    assert predicted.grad[2] > 0


def test_memory_equivalence_selector_is_exact_noop_with_live_credit() -> None:
    model = UnifiedCognitiveController(
        width=32, workspace_slots=4, intention_width=8,
        adaptive_memory_usage_prior=True,
        adaptive_memory_usage_prior_hidden=8,
        adaptive_memory_usage_prior_proposer_hidden=8,
        adaptive_memory_equivalence_hidden=8)
    features = torch.zeros(6, 12)
    features[:, 4:8] = torch.tensor([1.0, 0.9, 0.8, 0.7])
    features[:, 8:12] = torch.tensor([0.2, 0.3, 0.5, 0.9])
    probe = torch.randn(6, 32)
    rows = torch.randn(6, 4, 32)
    inherited = model.memory_usage_prior_probability(features)
    expanded = model.memory_equivalence_probability(features, probe, rows)
    assert torch.equal(expanded, inherited)
    logits = model.memory_equivalence_logits(probe, rows)
    loss = torch.nn.functional.cross_entropy(
        logits, torch.arange(6) % 4)
    loss.backward()
    assert model.memory_equivalence_selector is not None
    assert model.memory_equivalence_selector[-1].weight.grad is not None
    assert bool(
        (model.memory_equivalence_selector[-1].weight.grad != 0).any())


def test_memory_equivalence_calibration_starts_as_identity() -> None:
    model = UnifiedCognitiveController(
        width=32, workspace_slots=4, intention_width=8,
        adaptive_memory_equivalence_hidden=8,
        adaptive_memory_equivalence_calibration=True)
    probe = torch.randn(6, 32)
    rows = torch.randn(6, 3, 32)
    raw = model.memory_equivalence_logits(probe, rows)
    calibrated = model.calibrated_memory_equivalence_logits(probe, rows)
    assert torch.equal(calibrated, raw)
    calibrated.sum().backward()
    assert model.memory_equivalence_logit_scale is not None
    assert model.memory_equivalence_logit_bias is not None
    assert model.memory_equivalence_logit_scale.grad is not None
    assert model.memory_equivalence_logit_bias.grad is not None


def test_representative_read_critic_is_separate_and_trainable() -> None:
    model = UnifiedCognitiveController(
        width=32, workspace_slots=4, intention_width=8,
        adaptive_representative_read_hidden=16)
    frames = torch.rand(4, 3, 32, 32)
    state = model.initial_state(4, device=torch.device("cpu"))
    action = torch.full((4,), 2, dtype=torch.long)
    zeros = torch.zeros(4)
    before, _ = model.step(
        frames, state, action, zeros, zeros)
    features = torch.randn(4, 32 * 5 + 4)
    probability = model.representative_deep_read_probability(features)
    assert probability.shape == (4,)
    probability.sum().backward()
    assert model.representative_read_critic is not None
    assert any(
        parameter.grad is not None
        for parameter in model.representative_read_critic.parameters())
    after, _ = model.step(
        frames, state, action, zeros, zeros)
    assert torch.equal(before.logits, after.logits)
    assert torch.equal(before.memory_value, after.memory_value)


def test_representative_read_examples_use_exact_interaction_budget() -> None:
    from .train_adaptive_representative_read import (
        representative_read_examples,
    )

    payload = torch.load(
        "artifacts/checkpoints/"
        "unified_equivalence_consolidation_seed20541.pt",
        map_location="cpu", weights_only=False)
    configuration = dict(payload["model_configuration"])
    configuration["adaptive_representative_read_hidden"] = 16
    model = UnifiedCognitiveController(**configuration)
    model.load_state_dict(payload["state_dict"], strict=False)
    examples = representative_read_examples(
        model, examples=6, seed=20700,
        reverse_rules=False, corrupt_memory=False,
        device=torch.device("cpu"))
    assert examples["features"].shape == (6, model.width * 5 + 4)
    assert examples["outcomes"].shape == (6, 2)
    assert examples["comparisons"].shape == (6, 2)
    assert set(examples["appearance_ids"].tolist()) == {0, 1, 2}


def test_natural_equivalence_batch_uses_only_mixed_behavioral_banks() -> None:
    from .train_natural_memory_equivalence import natural_equivalence_batch

    payload = torch.load(
        "artifacts/checkpoints/"
        "unified_four_target_shape_transfer_seed19511.pt",
        map_location="cpu", weights_only=False)
    configuration = dict(payload["model_configuration"])
    configuration["adaptive_memory_equivalence_hidden"] = 8
    model = UnifiedCognitiveController(**configuration)
    model.load_state_dict(payload["state_dict"], strict=False)
    data = natural_equivalence_batch(
        model, count=16, seed=20200, device=torch.device("cpu"),
        heldout=True, exact_fraction=0.0)
    assert data["sorted_values"].shape == (16, 4, model.width)
    assert data["sorted_outcomes"].shape == (16, 4)
    assert bool((data["duplicate_count"] >= 1).all())
    assert bool((data["duplicate_count"] <= 3).all())
    assert data["mining_verifier_bits"] == 16 * 2 * 4
    assert data["generated_contexts"] == 16 * 2 * 5


def test_natural_equivalence_counterfactual_preserves_bank_and_flips_outcomes(
        ) -> None:
    from .train_natural_memory_equivalence import natural_equivalence_batch

    payload = torch.load(
        "artifacts/checkpoints/"
        "unified_four_target_shape_transfer_seed19511.pt",
        map_location="cpu", weights_only=False)
    configuration = dict(payload["model_configuration"])
    configuration["adaptive_memory_equivalence_hidden"] = 8
    model = UnifiedCognitiveController(**configuration)
    model.load_state_dict(payload["state_dict"], strict=False)
    common = {
        "model": model,
        "count": 16,
        "seed": 20209,
        "device": torch.device("cpu"),
        "heldout": True,
        "exact_fraction": 0.0,
    }
    ordinary = natural_equivalence_batch(**common)
    reversed_data = natural_equivalence_batch(
        **common, reverse_target_rule=True)
    for name in ("keys", "values", "usage", "queries", "sorted_values"):
        assert torch.equal(ordinary[name], reversed_data[name])
    assert torch.equal(
        reversed_data["sorted_outcomes"],
        1.0 - ordinary["sorted_outcomes"])
    assert bool(
        (ordinary["probe_values"] != reversed_data["probe_values"])
        .any(-1).all())


def test_natural_equivalence_relation_is_physical_row_invariant() -> None:
    from .train_four_target_memory_retrieval import policy_features
    from .train_natural_memory_equivalence import natural_equivalence_batch

    payload = torch.load(
        "artifacts/checkpoints/"
        "unified_four_target_shape_transfer_seed19511.pt",
        map_location="cpu", weights_only=False)
    configuration = dict(payload["model_configuration"])
    configuration["adaptive_memory_equivalence_hidden"] = 8
    model = UnifiedCognitiveController(**configuration)
    model.load_state_dict(payload["state_dict"], strict=False)
    common = {
        "model": model,
        "count": 16,
        "seed": 20210,
        "device": torch.device("cpu"),
        "heldout": True,
        "exact_fraction": 0.0,
    }
    permuted = natural_equivalence_batch(**common, permute_rows=True)
    ordered = natural_equivalence_batch(**common, permute_rows=False)
    assert torch.equal(permuted["sorted_values"], ordered["sorted_values"])
    assert torch.equal(permuted["sorted_outcomes"], ordered["sorted_outcomes"])
    assert torch.equal(
        policy_features(permuted), policy_features(ordered))


def test_nearest_verified_candidate_loss_preserves_disconnected_modes() -> None:
    from .train_natural_memory_equivalence import (
        nearest_verified_candidate_loss,
    )

    predicted = torch.tensor([0.2, 0.8, 0.5], requires_grad=True)
    candidates = torch.tensor([
        [0.1, 0.4, 0.7, 0.9],
        [0.1, 0.4, 0.7, 0.9],
        [0.1, 0.4, 0.7, 0.9],
    ])
    successful = torch.tensor([
        [True, False, False, True],
        [True, False, False, True],
        [True, False, False, True],
    ])
    loss = nearest_verified_candidate_loss(
        predicted, candidates, successful)
    assert loss.item() == pytest.approx((0.01 + 0.01 + 0.16) / 3)
    loss.backward()
    assert predicted.grad is not None


def test_equivalence_consolidation_never_exceeds_capacity() -> None:
    from .train_equivalence_consolidation import (
        consolidate,
        natural_memory_streams,
    )

    payload = torch.load(
        "artifacts/checkpoints/"
        "unified_natural_memory_equivalence_seed20252.pt",
        map_location="cpu", weights_only=False)
    configuration = dict(payload["model_configuration"])
    configuration["adaptive_memory_equivalence_calibration"] = True
    model = UnifiedCognitiveController(**configuration)
    model.load_state_dict(payload["state_dict"], strict=False)
    data = natural_memory_streams(
        model, streams=8, length=8, seed=20500,
        device=torch.device("cpu"), heldout=True)
    bank = consolidate(model, data, capacity=2, policy="learned")
    assert bank["values"].shape == (8, 2, model.width)
    assert bool((bank["valid"].sum(-1) <= 2).all())
    assert bool((bank["usage"] >= 0).all())


def test_equivalence_consolidation_can_reserve_in_class_diversity() -> None:
    from .train_equivalence_consolidation import (
        consolidate,
        natural_memory_streams,
    )

    payload = torch.load(
        "artifacts/checkpoints/"
        "unified_equivalence_consolidation_seed20541.pt",
        map_location="cpu", weights_only=False)
    model = UnifiedCognitiveController(**payload["model_configuration"])
    model.load_state_dict(payload["state_dict"])
    data = natural_memory_streams(
        model, streams=8, length=16, seed=20560,
        device=torch.device("cpu"), heldout=True)
    compact = consolidate(
        model, data, capacity=2, representatives_per_class=1)
    diverse = consolidate(
        model, data, capacity=4, representatives_per_class=2)
    assert bool((compact["valid"].sum(-1) <= 2).all())
    assert bool((diverse["valid"].sum(-1) <= 4).all())
    assert float(diverse["valid"].sum(-1).float().mean()) > float(
        compact["valid"].sum(-1).float().mean())
    assert bool((diverse["cluster_ids"][diverse["valid"]] >= 0).all())
    assert bool(
        (diverse["representative_ranks"][diverse["valid"]] >= 0).all())
    for stream in range(8):
        for cluster in diverse["cluster_ids"][stream].unique():
            if int(cluster) < 0:
                continue
            selected = diverse["cluster_ids"][stream] == cluster
            ranks = diverse["representative_ranks"][stream, selected]
            assert torch.equal(
                ranks.sort().values,
                torch.arange(ranks.numel(), dtype=torch.long))


def test_usage_prior_residual_is_exact_noop_at_insertion() -> None:
    payload = torch.load(
        "artifacts/checkpoints/"
        "unified_four_target_memory_retrieval_seed17828.pt",
        map_location="cpu", weights_only=False)
    parent = UnifiedCognitiveController(**payload["model_configuration"])
    parent.load_state_dict(payload["state_dict"])
    configuration = dict(payload["model_configuration"])
    configuration["adaptive_memory_usage_prior_residual_hidden"] = 8
    configuration["adaptive_memory_usage_prior_residual_features"] = 12
    expanded = UnifiedCognitiveController(**configuration)
    missing, unexpected = expanded.load_state_dict(
        payload["state_dict"], strict=False)
    assert set(missing) == {
        "memory_usage_prior_residual.0.weight",
        "memory_usage_prior_residual.0.bias",
        "memory_usage_prior_residual.2.weight",
        "memory_usage_prior_residual.2.bias",
    }
    assert not unexpected
    features = torch.randn(64, 4)
    assert torch.equal(
        parent.memory_usage_prior_probability(features),
        expanded.memory_usage_prior_probability(features))
    rich_features = torch.randn(64, 12)
    rich_features[:, :4] = features
    assert torch.equal(
        parent.memory_usage_prior_probability(features),
        expanded.memory_usage_prior_probability(rich_features))


def test_usage_prior_relational_proposer_is_exact_noop_with_live_gradient() -> None:
    payload = torch.load(
        "artifacts/checkpoints/"
        "unified_four_target_boundary_transfer_seed17915.pt",
        map_location="cpu", weights_only=False)
    parent = UnifiedCognitiveController(**payload["model_configuration"])
    parent.load_state_dict(payload["state_dict"])
    configuration = dict(payload["model_configuration"])
    configuration["adaptive_memory_usage_prior_proposer_hidden"] = 16
    expanded = UnifiedCognitiveController(**configuration)
    missing, unexpected = expanded.load_state_dict(
        payload["state_dict"], strict=False)
    assert set(missing) == {
        "memory_usage_prior_proposer.0.weight",
        "memory_usage_prior_proposer.0.bias",
        "memory_usage_prior_proposer.2.weight",
        "memory_usage_prior_proposer.2.bias",
    }
    assert not unexpected
    features = torch.randn(64, 12)
    features[:, 8:12] = torch.tensor([0.2, 0.4, 0.7, 1.0])
    inherited = parent.memory_usage_prior_probability(features)
    proposed = expanded.memory_usage_prior_probability(features)
    assert torch.equal(inherited, proposed)
    proposed.mean().backward()
    assert expanded.memory_usage_prior_proposer is not None
    output = expanded.memory_usage_prior_proposer[-1]
    assert output.weight.grad is not None
    assert output.weight.grad[4].abs().sum() > 0
    expanded.zero_grad(set_to_none=True)
    with torch.no_grad():
        output.bias[4] = -1.0
    assert torch.equal(
        inherited, expanded.memory_usage_prior_probability(features))
    expanded.memory_usage_prior_probability(features).mean().backward()
    assert output.bias.grad is not None
    assert output.bias.grad[4].abs() > 0


def test_full_memory_usage_features_are_row_permutation_invariant() -> None:
    base = torch.randn(8, 4)
    queries = torch.randn(8, 16)
    keys = torch.randn(8, 4, 16)
    usage = torch.rand(8, 4)
    permutation = torch.rand(8, 4).argsort(-1)
    gather_keys = permutation.unsqueeze(-1).expand(-1, -1, 16)
    expected = full_memory_usage_features(base, queries, keys, usage)
    actual = full_memory_usage_features(
        base, queries,
        torch.gather(keys, 1, gather_keys),
        torch.gather(usage, 1, permutation))
    assert torch.equal(expected, actual)


def test_persistent_stream_age_uses_current_insertion_rank() -> None:
    memory = DiskLatentMemory(width=4, capacity=3)
    memory.commit(
        torch.eye(4)[:3], torch.eye(4)[:3],
        torch.ones(3), threshold=0.0)
    memory.store.age[:3] = torch.tensor([17, 4, 11])
    assert torch.allclose(
        _ranked_age(memory),
        torch.tensor([1.0, 1 / 3, 2 / 3]))


def test_persistent_stream_future_replaces_only_selected_rows() -> None:
    batch = generate_lifetimes(4, 3, seed=330)
    candidate = generate_lifetimes(2, 3, seed=332)
    queries = torch.arange(16, dtype=torch.float32).reshape(2, 2, 4)
    candidate_queries = torch.full((2, 4), -1.0)
    updated, updated_queries = _future_for_actions(
        batch, queries, candidate, candidate_queries,
        torch.tensor([0, 2]), capacity=2)
    assert torch.equal(updated.seeds[:2], batch.seeds[:2])
    assert torch.equal(updated.seeds[2], batch.seeds[2])
    assert torch.equal(updated.seeds[3], candidate.seeds[1])
    assert torch.equal(updated_queries[0], queries[0])
    assert torch.equal(updated_queries[1, 0], queries[1, 0])
    assert torch.equal(updated_queries[1, 1], candidate_queries[1])


def test_efficiency_metric_normalizes_against_same_state_frozen_policy() -> None:
    report = {
        "total_seconds": 1.5,
        "trace": [
            {
                "phase": "reliability_dominant",
                "learned_reward": 0.8,
                "frozen_reward": 0.6,
                "learned_target_rate": 0.7,
                "frozen_target_rate": 0.4,
            },
            {
                "phase": "old_return",
                "learned_reward": 0.9,
                "frozen_reward": 0.85,
                "learned_target_rate": 0.8,
                "frozen_target_rate": 0.75,
            },
        ],
    }
    metrics = _arm_metrics(report)
    assert abs(metrics["verified_reward_advantage_auc"] - 0.25) < 1e-6
    assert abs(metrics["target_rate_advantage_auc"] - 0.35) < 1e-6
    assert abs(metrics["old_return_reward_advantage"] - 0.05) < 1e-6


def test_strategy_memory_is_bounded_retrievable_and_persistent(
        tmp_path: Path) -> None:
    memory = LatentStrategyMemory(
        capacity=2, key_width=3, value_width=2)
    keys = torch.eye(3)
    memory.upsert(
        keys[0], torch.tensor([1.0, 0.0]),
        verified_improvement=0.2)
    memory.upsert(
        keys[1], torch.tensor([0.0, 1.0]),
        verified_improvement=-0.1)
    retrieved = memory.retrieve(
        keys[0], torch.zeros(2))
    assert retrieved.slot == 0
    assert torch.equal(retrieved.value, torch.tensor([1.0, 0.0]))
    memory.upsert(
        keys[2], torch.tensor([-1.0, -1.0]),
        verified_improvement=0.3)
    assert memory.count == 2
    path = tmp_path / "strategy-memory.pt"
    memory.save(path)
    restored = LatentStrategyMemory.load(path)
    assert restored.count == 2
    assert torch.equal(restored.keys, memory.keys)
    assert torch.equal(restored.values, memory.values)
    assert torch.equal(restored.usage, memory.usage)
    assert torch.equal(restored.success, memory.success)
    assert torch.equal(restored.failure, memory.failure)


def test_strategy_memory_can_replace_a_preferred_duplicate_slot() -> None:
    memory = LatentStrategyMemory(
        capacity=2, key_width=2, value_width=1)
    memory.upsert(
        torch.tensor([1.0, 0.0]), torch.tensor([1.0]),
        verified_improvement=1.0)
    memory.upsert(
        torch.tensor([0.0, 1.0]), torch.tensor([2.0]),
        verified_improvement=1.0)
    slot = memory.upsert(
        torch.tensor([-1.0, 0.0]), torch.tensor([3.0]),
        verified_improvement=1.0, preferred_slot=0)
    assert slot == 0
    assert memory.count == 2
    assert torch.equal(memory.values[:, 0], torch.tensor([3.0, 2.0]))


def test_skill_adapter_can_read_only_the_immediately_prior_slot() -> None:
    model = UnifiedCognitiveController(
        width=16,
        intention_width=4,
        skill_adapter_widths=(5, 7, 11),
        skill_adapter_reads_prior=True,
        skill_adapter_prior_read_limit=1,
    )
    assert model.skill_adapters[0][0].in_features == 32
    assert model.skill_adapters[1][0].in_features == 32 + 5
    assert model.skill_adapters[2][0].in_features == 32 + 7


def test_zero_prior_read_limit_keeps_all_earlier_slots() -> None:
    model = UnifiedCognitiveController(
        width=16,
        intention_width=4,
        skill_adapter_widths=(5, 7, 11),
        skill_adapter_reads_prior=True,
        skill_adapter_prior_read_limit=0,
    )
    assert model.skill_adapters[2][0].in_features == 32 + 5 + 7


def test_prior_slot_volatility_names_only_recent_inherited_slots() -> None:
    assert _prior_slot_prefixes(3, 1) == (
        "skill_adapters.2.",
        "skill_adapter_gates.2.",
        "skill_adapter_read_projections.2.",
    )
    assert _prior_slot_prefixes(3, 2) == (
        "skill_adapters.1.",
        "skill_adapter_gates.1.",
        "skill_adapter_read_projections.1.",
        "skill_adapters.2.",
        "skill_adapter_gates.2.",
        "skill_adapter_read_projections.2.",
    )


def test_relative_state_drift_is_zero_then_detects_change() -> None:
    model = torch.nn.Linear(2, 1, bias=False)
    initial = {"weight": model.weight.detach().cpu().clone()}
    assert _relative_state_drift(model, initial) == 0.0
    model.weight.data.add_(0.25)
    assert _relative_state_drift(model, initial) > 0.0


def test_alignment_volatility_protects_conflicts_and_thaws_agreement() -> None:
    assert _alignment_volatility(-1.0, 0.2) < 0.001
    assert _alignment_volatility(0.0, 0.2) == pytest.approx(0.1)
    assert _alignment_volatility(1.0, 0.2) > 0.199


def test_gradient_alignment_identifies_agreement_and_conflict() -> None:
    parameter = torch.nn.Parameter(torch.tensor([1.0, -1.0]))
    positive = (parameter.square()).sum()
    aligned = (parameter * 2).square().sum()
    opposed = -aligned
    assert _gradient_alignment(positive, aligned, [parameter]) > 0.99
    assert _gradient_alignment(positive, opposed, [parameter]) < -0.99


def test_importance_volatility_protects_used_hidden_units() -> None:
    volatility = _importance_volatility(
        torch.tensor([0.0, 1.0, 4.0]), maximum=0.2, strength=4.0)
    assert volatility[0] == pytest.approx(0.2)
    assert volatility[0] > volatility[1] > volatility[2]


def test_unit_update_blending_applies_one_scalar_per_output_unit() -> None:
    parameter = torch.nn.Parameter(torch.tensor([
        [2.0, 4.0],
        [6.0, 8.0],
    ]))
    before = torch.tensor([
        [1.0, 1.0],
        [2.0, 2.0],
    ])
    _blend_unit_update(
        parameter, before, torch.tensor([0.0, 0.5]))
    assert torch.equal(parameter[0], before[0])
    assert torch.equal(parameter[1], torch.tensor([4.0, 5.0]))


def test_physical_context_key_ignores_skip_and_is_normalized() -> None:
    features = torch.zeros(2, 4, 7)
    features[:, 1:, 0] = 0.5
    features[:, 1:, 5] = -0.25
    features[:, 1:, 6] = 0.25
    first = physical_context_key(features)
    features[:, 0] = 999
    second = physical_context_key(features)
    assert torch.allclose(first, second)
    assert torch.allclose(first.norm(), torch.tensor(1.0))
    rewarded = physical_context_key(
        features, torch.tensor([0.4, 0.2, 0.1]))
    assert rewarded.shape == (13,)
    assert torch.allclose(rewarded.norm(), torch.tensor(1.0))


def test_context_encoder_learns_only_from_verified_improvement() -> None:
    encoder = VerifierTrainedContextEncoder(width=3)
    optimizer = torch.optim.SGD(encoder.parameters(), lr=0.5)
    keys = torch.tensor([[1.0, 1.0, 0.0], [1.0, -1.0, 0.0]])
    query = torch.tensor([1.0, 1.0, 0.0])
    before = encoder.log_scale.detach().clone()
    loss = encoder.reinforce(
        query, keys, selected_slot=0, verified_improvement=1.0,
        optimizer=optimizer)
    assert loss > 0
    assert not torch.equal(encoder.log_scale, before)
    learned = encoder.log_scale.detach().clone()
    encoder.reinforce(
        query, keys, selected_slot=0, verified_improvement=0.0,
        optimizer=optimizer)
    assert torch.equal(encoder.log_scale, learned)


def test_strategy_memory_can_retrieve_with_learned_context_metric() -> None:
    memory = LatentStrategyMemory(
        capacity=2, key_width=3, value_width=1)
    memory.upsert(
        torch.tensor([1.0, 0.1, 0.0]), torch.tensor([1.0]),
        verified_improvement=1.0)
    memory.upsert(
        torch.tensor([0.1, 1.0, 0.0]), torch.tensor([2.0]),
        verified_improvement=1.0)
    encoder = VerifierTrainedContextEncoder(width=3)
    encoder.log_scale.data.copy_(torch.tensor([3.0, -3.0, 0.0]))
    result = memory.retrieve(
        torch.tensor([0.2, 1.0, 0.0]), torch.zeros(1),
        encoder=encoder)
    assert result.slot == 0


def test_soft_strategy_retrieval_is_a_convex_mixture() -> None:
    memory = LatentStrategyMemory(
        capacity=2, key_width=2, value_width=2)
    memory.upsert(
        torch.tensor([1.0, 0.0]), torch.tensor([2.0, 0.0]),
        verified_improvement=1.0)
    memory.upsert(
        torch.tensor([0.0, 1.0]), torch.tensor([0.0, 4.0]),
        verified_improvement=1.0)
    encoder = VerifierTrainedContextEncoder(width=2)
    result = memory.retrieve_soft(
        torch.tensor([1.0, 1.0]), torch.zeros(2),
        encoder=encoder, temperature=0.5)
    assert result.mixture_weights is not None
    assert torch.allclose(
        result.mixture_weights, torch.tensor([0.5, 0.5]))
    assert torch.allclose(result.value, torch.tensor([1.0, 2.0]))


def test_context_encoder_spsa_step_follows_verified_difference() -> None:
    encoder = VerifierTrainedContextEncoder(width=3)
    direction = torch.tensor([1.0, -1.0, 1.0])
    advantage = encoder.spsa_step(
        direction, positive_reward=0.8, negative_reward=0.3,
        step_size=0.2)
    assert abs(advantage - 0.5) < 1e-6
    assert torch.allclose(
        encoder.log_scale,
        torch.tensor([0.1, -0.1, 0.1]))


def test_context_reliability_ramp_has_six_rounds_and_one_changing_axis(
        ) -> None:
    phases = _curriculum_phases(
        "context_reliability_ramp", rounds_per_phase=99)
    assert len(phases) == 6
    assert all(rounds == 1 for _, _, rounds in phases)
    reliability = [weights[2] for _, weights, _ in phases]
    assert reliability == [0.0, 0.1, 0.2, 0.3, 0.4, 0.0]
    for _, weights, _ in phases:
        assert abs(sum(weights) - 1.0) < 1e-9
        assert weights[0] == weights[1]


def test_interleaved_reliability_exposes_every_context_in_each_short_cycle(
        ) -> None:
    phases = _curriculum_phases(
        "interleaved_reliability", rounds_per_phase=4)
    assert len(phases) == 12
    assert all(rounds == 1 for _, _, rounds in phases)
    assert [phase for phase, _, _ in phases] == [
        "old_equal", "reliability_dominant", "old_return",
    ] * 4
    assert [weights for _, weights, _ in phases[:3]] == [
        (0.5, 0.5, 0.0),
        (0.3, 0.3, 0.4),
        (0.5, 0.5, 0.0),
    ]


def test_cyclic_reliability_blocks6_preserves_six_round_context_runs(
        ) -> None:
    phases = _curriculum_phases(
        "cyclic_reliability_blocks6", rounds_per_phase=18)
    assert [phase for phase, _, _ in phases] == [
        "old_equal", "reliability_dominant", "old_return",
    ] * 3
    assert [rounds for _, _, rounds in phases] == [6] * 9
    assert sum(rounds for _, _, rounds in phases) == 54


def test_cyclic_reliability_blocks6_rejects_incompatible_budget() -> None:
    with pytest.raises(ValueError, match="divisible by six"):
        _curriculum_phases("cyclic_reliability_blocks6", rounds_per_phase=5)


def test_value_diverse_admission_preserves_latent_extremes() -> None:
    memory = LatentStrategyMemory(
        capacity=2, key_width=2, value_width=2)
    memory.upsert(
        torch.tensor([1.0, 0.0]), torch.tensor([-2.0, 0.0]),
        verified_improvement=1.0)
    memory.upsert(
        torch.tensor([0.0, 1.0]), torch.tensor([0.0, 0.0]),
        verified_improvement=1.0)
    candidate, slot = _value_diverse_admission(
        memory,
        [
            torch.tensor([[0.1, 0.0]]),
            torch.tensor([[3.0, 0.0]]),
        ],
        [1.0, 0.5])
    assert candidate == 1
    assert slot == 1


def test_prefix_state_strategy_memory_round_trip_is_exact() -> None:
    source = LatentStrategyMemory(
        capacity=3, key_width=2, value_width=2)
    source.upsert(
        torch.tensor([1.0, 0.0]), torch.tensor([-2.0, 0.5]),
        verified_improvement=1.0)
    source.upsert(
        torch.tensor([0.0, 1.0]), torch.tensor([3.0, -0.5]),
        verified_improvement=-1.0)
    source.usage[:2] = torch.tensor([7, 4])
    restored = _restore_strategy_memory(
        _strategy_memory_payload(source), device=torch.device("cpu"))
    assert restored is not None
    assert restored.count == source.count
    for field in ("keys", "values", "usage", "success", "failure"):
        assert torch.equal(getattr(restored, field), getattr(source, field))


def test_dynamic_working_memory_mask_accounting_and_persistence(
        tmp_path: Path) -> None:
    memory = DynamicWorkingMemory(
        capacity=8, width=3, fixed_active_slots=4)
    assert memory.active_count == 4
    memory.write(1, torch.tensor([1.0, 2.0, 3.0]))
    assert torch.equal(memory.read(1), torch.tensor([1.0, 2.0, 3.0]))
    memory.write(4, torch.zeros(3))
    memory.write(4, torch.zeros(3))
    assert memory.stats.evictions == 1
    selected = memory.set_active_from_scores(
        torch.tensor([-1.0, -1.0, 0.9, 0.8, -1.0, -1.0, -1.0, -1.0]),
        minimum=2)
    assert selected.tolist() == [
        False, False, True, True, False, False, False, False]
    memory.record_step()
    assert memory.stats.mean_occupancy == 0.25
    path = tmp_path / "dynamic-memory.pt"
    memory.save(path)
    restored = DynamicWorkingMemory.load(path)
    assert torch.equal(restored.values, memory.values)
    assert torch.equal(restored.active, memory.active)
    assert torch.equal(restored.occupied, memory.occupied)
    assert torch.equal(restored.usage, memory.usage)
    assert torch.equal(restored.age, memory.age)
    assert restored.stats == memory.stats


def test_capability_ledger_records_memory_and_latency() -> None:
    memory = DynamicWorkingMemory(
        capacity=8, width=2, fixed_active_slots=2)
    memory.record_step()
    memory.active_values()
    ledger = CapabilityLedger(
        unique_verifier_bits=16, unique_logical_lifetimes=4)
    ledger.absorb_memory(memory.stats)
    with LatencyTimer(ledger):
        _ = sum(range(10))
    report = ledger.as_report()
    assert report["memory_reads"] == 2
    assert report["mean_active_fraction"] == 0.25
    assert report["latency_seconds"] >= 0.0


def test_old_disk_schema_loads_with_zero_access_counts(tmp_path: Path) -> None:
    path = tmp_path / "old-memory.pt"
    torch.save({
        "schema": "syllogimous-neural-computer-memory-v1",
        "keys": torch.zeros(2, 4),
        "values": torch.zeros(2, 4),
        "usage": torch.zeros(2),
        "age": torch.zeros(2, dtype=torch.long),
        "valid": torch.tensor([True, False]),
        "clock": 1,
        "growth_chunk": 2,
    }, path)
    restored = DiskLatentMemory.load(path)
    assert restored.count == 1
    assert restored.store.access_count.tolist() == [0, 0]
    assert restored.store.success_count.tolist() == [0, 0]
    assert restored.store.failure_count.tolist() == [0, 0]


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


def test_adaptive_memory_replacement_is_optional_and_bounded() -> None:
    ordinary = UnifiedCognitiveController(
        width=32, workspace_slots=4, intention_width=8)
    assert ordinary.memory_replacement_gate is None
    model = UnifiedCognitiveController(
        width=32, workspace_slots=4, intention_width=8,
        adaptive_memory_replace=True,
        adaptive_memory_replace_hidden=8)
    scores = model.memory_replacement_scores(torch.zeros(3, 5, 5))
    assert scores.shape == (3, 5)
    assert sum(
        parameter.numel()
        for parameter in model.memory_replacement_gate.parameters()) == 57
    frequency_model = UnifiedCognitiveController(
        width=32, workspace_slots=4, intention_width=8,
        adaptive_memory_replace=True,
        adaptive_memory_replace_hidden=8,
        adaptive_memory_replace_features=6)
    frequency_scores = frequency_model.memory_replacement_scores(
        torch.zeros(3, 5, 6))
    assert frequency_scores.shape == (3, 5)
    assert sum(
        parameter.numel()
        for name, parameter in frequency_model.named_parameters()
        if name.startswith("memory_replacement_")) == 58
    missing, unexpected = frequency_model.load_state_dict(
        model.state_dict(), strict=False)
    assert missing == ["memory_replacement_extra_gate.weight"]
    assert not unexpected
    base_features = torch.randn(3, 5, 5)
    expanded_features = torch.cat((
        base_features, torch.zeros(3, 5, 1)), dim=-1)
    assert torch.equal(
        model.memory_replacement_scores(base_features),
        frequency_model.memory_replacement_scores(expanded_features))


def test_frequency_recency_utility_weights_are_validated_before_generation(
        ) -> None:
    model = UnifiedCognitiveController(
        width=32, workspace_slots=4, intention_width=8)
    for recency_weight, frequency_weight in (
            (-0.1, 1.0), (1.0, -0.1), (0.0, 0.0)):
        try:
            frequency_recency_batch(
                model, banks=1, capacity=2, seed=1,
                device=torch.device("cpu"), write_threshold=0.5,
                recency_weight=recency_weight,
                frequency_weight=frequency_weight)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid utility weights were accepted")
    try:
        frequency_recency_batch(
            model, banks=1, capacity=2, seed=1,
            device=torch.device("cpu"), write_threshold=0.5,
            reliability_weight=-0.1)
    except ValueError:
        pass
    else:
        raise AssertionError("negative reliability weight was accepted")


def test_eight_feature_expansion_preserves_old_replacement_scores() -> None:
    source = UnifiedCognitiveController(
        width=32, workspace_slots=4, intention_width=8,
        adaptive_memory_read=True,
        adaptive_memory_replace=True,
        adaptive_memory_replace_hidden=8,
        adaptive_memory_replace_features=6)
    source.memory_replacement_extra_gate.weight.data.fill_(0.75)
    configuration = {
        "width": 32,
        "workspace_slots": 4,
        "intention_width": 8,
        "adaptive_memory_read": True,
        "adaptive_memory_replace": True,
        "adaptive_memory_replace_hidden": 8,
        "adaptive_memory_replace_features": 6,
    }
    expanded = _expanded_eight_feature_controller(
        configuration, source.state_dict(), device=torch.device("cpu"))
    old_features = torch.randn(3, 5, 6)
    new_features = torch.cat((
        old_features, torch.zeros(3, 5, 2)), dim=-1)
    assert torch.equal(
        source.memory_replacement_scores(old_features),
        expanded.memory_replacement_scores(new_features))
    assert torch.equal(
        expanded.memory_replacement_extra_gate.weight,
        torch.tensor([[0.75, 0.0, 0.0]]))


def test_full_replacement_policy_reset_preserves_controller_backbone(
        ) -> None:
    source = UnifiedCognitiveController(
        width=32, workspace_slots=4, intention_width=8,
        adaptive_memory_read=True,
        adaptive_memory_replace=True,
        adaptive_memory_replace_hidden=8,
        adaptive_memory_replace_features=6)
    configuration = {
        "width": 32,
        "workspace_slots": 4,
        "intention_width": 8,
        "adaptive_memory_read": True,
        "adaptive_memory_replace": True,
        "adaptive_memory_replace_hidden": 8,
        "adaptive_memory_replace_features": 6,
    }
    arms = build_transfer_arms(
        {
            "schema": "unified-cognitive-controller-v1",
            "model_configuration": configuration,
            "state_dict": source.state_dict(),
        },
        {
            "schema": "unified-controller-physical-prefix-state-v1",
            "model_state_dict": source.state_dict(),
        },
        device=torch.device("cpu"), fresh_seed=7)
    reset = arms["replacement_policy_reset"]
    assert all(
        int(torch.count_nonzero(parameter)) == 0
        for parameter in reset.memory_replacement_gate.parameters())
    assert int(torch.count_nonzero(
        reset.memory_replacement_extra_gate.weight)) == 0
    assert torch.equal(
        reset.vision.network[0].weight,
        arms["selected_experience"].vision.network[0].weight)


def test_redundancy_batch_is_seed_exact_and_exposes_only_row_statistic(
        ) -> None:
    model = UnifiedCognitiveController(
        width=32, workspace_slots=4, intention_width=8,
        adaptive_memory_read=True,
        adaptive_memory_replace=True,
        adaptive_memory_replace_hidden=8,
        adaptive_memory_replace_features=8)
    kwargs = {
        "banks": 2,
        "capacity": 3,
        "seed": 73,
        "device": torch.device("cpu"),
        "write_threshold": 0.0,
        "noise_scale": 0.01,
        "weights": (0.3, 0.3, 0.3, 0.1),
    }
    first = redundancy_utility_batch(model, **kwargs)
    second = redundancy_utility_batch(model, **kwargs)
    assert first["option_features"].shape == (2, 4, 8)
    assert torch.equal(
        first["option_features"], second["option_features"])
    assert torch.equal(first["target_action"], second["target_action"])
    assert torch.equal(
        first["option_features"][:, 1:, 7],
        first["row_novelty"] - 0.5)
    assert torch.all(first["target_action"] >= 1)
    assert torch.all(first["target_action"] <= 3)


def test_bank_outcome_horizon_returns_only_requested_verifier_events() -> None:
    model = UnifiedCognitiveController(
        width=32, workspace_slots=4, intention_width=8,
        adaptive_memory_read=True,
        adaptive_memory_replace=True,
        adaptive_memory_replace_hidden=8,
        adaptive_memory_replace_features=8)
    data = redundancy_utility_batch(
        model, banks=4, capacity=3, seed=79,
        device=torch.device("cpu"), write_threshold=0.0,
        noise_scale=0.0, weights=(0.0, 0.0, 0.0, 1.0))
    actions = torch.tensor([0, 1, 2, 3])
    complete = _bank_outcomes(
        model, data, actions, device=torch.device("cpu"))
    immediate = _bank_outcomes(
        model, data, actions, device=torch.device("cpu"), horizon=1)
    assert complete.shape == (4, 3)
    assert immediate.shape == (4, 1)
    assert torch.equal(immediate[:, 0], complete[:, 0])
    assert torch.all((immediate == 0) | (immediate == 1))
    evidence = immediate_read_evidence(data, actions)
    assert evidence.shape == (4, 4)
    assert torch.isfinite(evidence).all()
    assert torch.equal(evidence[:, 3], torch.ones(4))


def test_stable_crossing_rejects_transient_threshold_hits() -> None:
    values = [(0, 0.4), (10, 0.6), (20, 0.5), (30, 0.7)]
    assert _stable_crossing(values, 0.55) == 30
    assert _stable_crossing(values, 0.75) is None


def test_saved_strategy_initializes_old_dimensions_only() -> None:
    model = UnifiedCognitiveController(
        width=32, workspace_slots=4, intention_width=8,
        adaptive_memory_read=True,
        adaptive_memory_replace=True,
        adaptive_memory_replace_hidden=8,
        adaptive_memory_replace_features=8)
    features = torch.zeros(2, 4, 8)
    features[:, 1:, 0] = 0.5
    selected = {
        "strategy_memory": {
            "count": 2,
            "keys": torch.stack((
                physical_context_key(
                    features, torch.tensor([0.8, 0.6, 0.4])),
                physical_context_key(
                    features, torch.tensor([0.4, 0.6, 0.8])),
            )),
            "values": torch.tensor([[2.0, -3.0], [-4.0, 5.0]]),
        },
        "context_encoder_state_dict": {
            "log_scale": torch.zeros(13),
        },
        "run_state": {
            "previous_reward_signature": torch.tensor([0.8, 0.6, 0.4]),
        },
    }
    report = initialize_from_saved_strategy(model, selected, features)
    assert report["slot"] == 0
    assert torch.equal(
        model.memory_replacement_extra_gate.weight,
        torch.tensor([[2.0, -3.0, 0.0]]))


def test_contextual_full_residual_is_exact_noop_until_retrieved() -> None:
    model = UnifiedCognitiveController(
        width=32, workspace_slots=4, intention_width=8,
        adaptive_memory_read=True,
        adaptive_memory_replace=True,
        adaptive_memory_replace_hidden=8,
        adaptive_memory_replace_features=8)
    features = torch.randn(2, 4, 8)
    key = full_residual_context_key(features)
    assert key.shape == (18,)
    assert torch.allclose(key.norm(), torch.tensor(1.0))
    memory = LatentStrategyMemory(
        capacity=2, key_width=18, value_width=8)
    residual, accepted, slot, similarity = retrieve_residual(
        memory, key, threshold=0.982)
    assert not accepted and slot is None and similarity == 0.0
    assert torch.count_nonzero(residual) == 0
    assert torch.equal(
        residual_scores(model, features, residual),
        model.memory_replacement_scores(features))


def test_contextual_full_residual_rejects_dissimilar_old_context() -> None:
    memory = LatentStrategyMemory(
        capacity=2, key_width=18, value_width=8)
    new_key = torch.nn.functional.normalize(
        torch.tensor([1.0] + [0.0] * 17), dim=0)
    old_key = torch.nn.functional.normalize(
        torch.tensor([0.0, 1.0] + [0.0] * 16), dim=0)
    value = torch.arange(8, dtype=torch.float32)
    memory.upsert(new_key, value, verified_improvement=1.0)
    accepted_value, accepted, slot, _ = retrieve_residual(
        memory, new_key, threshold=0.982)
    assert accepted and slot == 0
    assert torch.equal(accepted_value, value)
    rejected_value, accepted, slot, similarity = retrieve_residual(
        memory, old_key, threshold=0.982)
    assert not accepted and slot is None and similarity == 0.0
    assert torch.count_nonzero(rejected_value) == 0


def test_suppress_novelty_adapter_is_exact_noop_at_zero() -> None:
    model = UnifiedCognitiveController(
        width=32, workspace_slots=4, intention_width=8,
        adaptive_memory_read=True,
        adaptive_memory_replace=True,
        adaptive_memory_replace_hidden=8,
        adaptive_memory_replace_features=8)
    features = torch.randn(2, 4, 8)
    assert torch.equal(
        adapter_scores(
            model, features, torch.zeros(2),
            mode="suppress_novelty"),
        model.memory_replacement_scores(features))
    adapter = torch.tensor([-6.0, -4.0])
    scores = adapter_scores(
        model, features, adapter, mode="suppress_novelty")
    assert scores.shape == (2, 4)
    assert not torch.equal(
        scores, model.memory_replacement_scores(features))


def test_verified_candidate_selection_prefers_center_on_ties() -> None:
    assert select_verified_candidate(
        torch.tensor([0.8, 0.8, 0.8])) == 1
    assert select_verified_candidate(
        torch.tensor([0.8000001, 0.8, 0.7])) == 1
    assert select_verified_candidate(
        torch.tensor([0.82, 0.8, 0.7])) == 0
    assert select_verified_candidate(
        torch.tensor([0.7, 0.8, 0.83])) == 2


def test_disk_memory_can_replace_without_growing(tmp_path: Path) -> None:
    memory = DiskLatentMemory(width=4, capacity=2)
    keys = torch.eye(4)[:2]
    values = torch.arange(8, dtype=torch.float32).reshape(2, 4)
    assert memory.commit(
        keys, values, torch.ones(2), threshold=0.0) == 2
    replacement_key = torch.eye(4)[2]
    replacement_value = torch.full((4,), 7.0)
    memory.replace(0, replacement_key, replacement_value, 0.8)
    assert memory.count == 2
    path = tmp_path / "replaced.pt"
    memory.save(path)
    restored = DiskLatentMemory.load(path)
    read, _ = restored.retrieve(
        replacement_key.unsqueeze(0), top_k=1,
        confidence_mode="cosine")
    assert torch.allclose(read, replacement_value.unsqueeze(0))
    assert restored.count == 2


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
