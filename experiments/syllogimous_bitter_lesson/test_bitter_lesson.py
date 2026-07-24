import torch
import numpy as np

from experiments.syllogimous_latent_agent.data import (EpisodeDataset, collate_episodes,
                                                        generate_public_episode)

from .model import BitterLessonAgent, model_config, parameter_count
from .counterfactual import counterfactual_pair
from .structural_transfer import MixedStructuralDataset, generate_branched_episode
from .parity_transfer import ParityDataset, generate_parity_episode
from .choice_reaction import (ChoiceReactionDataset, CognitiveMixtureDataset,
                              ReactionDifficulty, generate_choice_reaction_episode)
from .choice_reaction_realtime import ChoiceReactionStream
from .cyclic_transfer import CyclicDataset, generate_cyclic_episode
from .train_rl import (direct_sensory_loss, policy_loss, q_learning_loss,
                       curriculum_sampling_choices, randomized_depth_verifier_loss,
                       reasoning_dataset, sample_halting, verifier_loss)


def tiny_batch():
    return collate_episodes([EpisodeDataset(2, premise_choices=(2,), entity_count=128)[i]
                             for i in range(2)])


def test_model_accepts_only_public_sensory_tensors():
    batch = tiny_batch()
    model = BitterLessonAgent(hidden=48, memory_slots=4, heads=4, max_thought_steps=3)
    output = model(batch["frames"], batch["pcm"], batch["mask"])
    assert output.observation_logits.shape == (2, 3, 5)
    assert output.answer_logits.shape == (2, 3, 5)
    assert output.halt_logits.shape == (2, 3)
    assert output.values.shape == (2, 3)


def test_all_generic_memory_cores_preserve_public_boundary():
    batch = tiny_batch()
    for core in ("soft_slots", "residual_slots", "residual_gru", "event_transformer"):
        model = BitterLessonAgent(hidden=48, memory_slots=4, heads=4,
                                  max_thought_steps=2, memory_core=core)
        output = model(batch["frames"], batch["pcm"], batch["mask"])
        assert output.answer_logits.shape == (2, 2, 5)


def test_gated_residual_thoughts_start_near_state_preserving():
    batch = tiny_batch()
    model = BitterLessonAgent(hidden=48, memory_slots=4, heads=4,
                              max_thought_steps=4, memory_core="event_transformer",
                              thought_dynamics="gated_residual")
    output = model(batch["frames"], batch["pcm"], batch["mask"])
    probabilities = output.answer_logits.softmax(-1)
    drift = (probabilities[:, 1:] - probabilities[:, :-1]).abs().mean()
    assert float(drift.detach()) < 0.01
    assert torch.allclose(model.thought_gate.bias,
                          torch.full_like(model.thought_gate.bias, -3.0))


def test_render_randomization_is_deterministic_and_changes_style():
    first = generate_public_episode(7, 2, entity_count=128, randomize_rendering=True)
    repeated = generate_public_episode(7, 2, entity_count=128, randomize_rendering=True)
    fixed = generate_public_episode(7, 2, entity_count=128, randomize_rendering=False)
    assert np.array_equal(first.frames, repeated.frames)
    assert not np.array_equal(first.frames, fixed.frames)
    assert np.array_equal(first.actions, fixed.actions)


def test_counterfactual_changes_only_conclusion_and_answer():
    original, counterfactual = counterfactual_pair(100_007, 3)
    assert np.array_equal(original.frames[:-1], counterfactual.frames[:-1])
    assert not np.array_equal(original.frames[-1], counterfactual.frames[-1])
    assert np.array_equal(original.pcm, counterfactual.pcm)
    assert np.array_equal(original.actions[:-1], counterfactual.actions[:-1])
    assert original.actions[-1] != counterfactual.actions[-1]


def test_branched_transfer_episode_is_deterministic_and_balanced():
    left = generate_branched_episode(300_007, 16, 4)
    right = generate_branched_episode(300_007, 16, 4)
    assert left.length == 17
    assert np.array_equal(left.frames, right.frames)
    assert left.actions[-1] in {2, 3}
    # Relevant endpoint Z00 appears in the conclusion and exactly one premise.
    assert int((left.subjects == 0).sum() + (left.objects == 0).sum()) == 2


def test_mixed_structural_dataset_alternates_public_task_shapes():
    dataset = MixedStructuralDataset(4, (2, 3), ((8, 2),))
    assert dataset[0].length in {3, 4}
    assert dataset[1].length == 9
    assert dataset[2].length in {3, 4}
    assert dataset[3].length == 9


def test_parity_episode_is_deterministic_and_balanced():
    first = generate_parity_episode(23, 6)
    repeated = generate_parity_episode(23, 6)
    assert first.length == 7
    assert np.array_equal(first.frames, repeated.frames)
    answers = [ParityDataset(200, (2, 3, 4))[index].actions[-1]
               for index in range(200)]
    true_fraction = sum(answer == 3 for answer in answers) / len(answers)
    assert 0.4 <= true_fraction <= 0.6


def test_cyclic_logic_increases_information_per_premise_deterministically():
    first = generate_cyclic_episode(31, 6, 8)
    repeated = generate_cyclic_episode(31, 6, 8)
    assert np.array_equal(first.frames, repeated.frames)
    assert np.array_equal(first.actions, repeated.actions)
    assert set(first.relations.tolist()).issubset(set(range(8)))
    answers = [CyclicDataset(400, (2, 4, 8), 8)[index].actions[-1]
               for index in range(400)]
    true_fraction = sum(answer == 3 for answer in answers) / len(answers)
    assert 0.45 <= true_fraction <= 0.55


def test_cyclic_logic_rejects_non_curriculum_moduli():
    try:
        generate_cyclic_episode(1, 2, 3)
    except ValueError as error:
        assert "modulus" in str(error)
    else:
        raise AssertionError("unsupported modulus should fail deterministically")


def test_one_premise_cyclic_perception_control_is_supported():
    episode = generate_cyclic_episode(9, 1, 4)
    assert episode.length == 2
    assert episode.actions[-1] in (2, 3)


def test_choice_reaction_is_deterministic_public_and_eight_way():
    difficulty = ReactionDifficulty(8, distractors=5, delay_frames=2,
                                    audio_distractors=3,
                                    target_like_distractors=4,
                                    temporal_distractors=2)
    first = generate_choice_reaction_episode(41, difficulty)
    repeated = generate_choice_reaction_episode(41, difficulty)
    assert first.length == 4
    assert first.group == 8
    assert 0 <= first.actions[-1] < 8
    assert np.array_equal(first.frames, repeated.frames)
    assert np.array_equal(first.pcm, repeated.pcm)
    assert np.all(first.actions[:-1] == 0)


def test_choice_curriculum_and_cognitive_mixture_collate():
    reactions = ChoiceReactionDataset(
        4, (ReactionDifficulty(2), ReactionDifficulty(4)))
    parity = ParityDataset(4, (2,))
    mixture = CognitiveMixtureDataset(parity, reactions)
    batch = collate_episodes([mixture[index] for index in range(4)])
    assert batch["frames"].shape[0] == 4
    assert batch["groups"].tolist() == [-1, 2, -1, 4]


def test_cognitive_mixture_supports_unequal_replay_ratios():
    reactions = ChoiceReactionDataset(2, (ReactionDifficulty(2),))
    parity = ParityDataset(6, (2,))
    mixture = CognitiveMixtureDataset(parity, reactions)
    items = [mixture[index] for index in range(len(mixture))]
    assert sum(item.group is not None for item in items) == 2
    assert sum(item.group is None for item in items) == 6


def test_mixed_reasoning_retains_parity_replay_at_requested_ratio():
    mixture = reasoning_dataset("mixed", 20, (2, 4), 4, 0.25,
                                start_seed=100, cyclic_lengths=(2,))
    assert len(mixture.reasoning) == 15
    assert len(mixture.reaction) == 5
    assert isinstance(mixture.reasoning, ParityDataset)
    assert isinstance(mixture.reaction, CyclicDataset)
    assert mixture.reasoning.premise_choices == (2, 4)
    assert mixture.reaction.premise_choices == (2,)


def test_eight_action_model_preserves_shared_policy_interface():
    batch = collate_episodes([
        generate_choice_reaction_episode(3, ReactionDifficulty(8)),
        generate_choice_reaction_episode(4, ReactionDifficulty(8)),
    ])
    model = BitterLessonAgent(hidden=48, memory_slots=4, heads=4,
                              max_thought_steps=2, action_count=8)
    output = model(batch["frames"], batch["pcm"], batch["mask"])
    assert output.answer_logits.shape == (2, 2, 8)


def test_realtime_stream_exposes_only_public_packets_and_verifies_latency():
    stream = ChoiceReactionStream(9, ReactionDifficulty(4, delay_frames=1),
                                  frame_interval_ms=1.0)
    packets = []
    while not stream.done_streaming:
        packets.append(stream.next_packet(realtime=False))
    assert len(packets) == 3
    assert [packet.stimulus_visible for packet in packets] == [False, False, True]
    target = generate_choice_reaction_episode(9, ReactionDifficulty(4, delay_frames=1),
                                              heldout=True).actions[-1]
    result = stream.verify(int(target), packets[-1].timestamp_ns,
                           deadline_ms=1000.0, speed_bonus=0.05)
    assert result.correct
    assert 1.0 <= result.reward <= 1.05


def test_reward_only_loss_is_finite_and_backpropagates():
    torch.manual_seed(3)
    batch = tiny_batch()
    model = BitterLessonAgent(hidden=48, memory_slots=4, heads=4, max_thought_steps=3)
    output = model(batch["frames"], batch["pcm"], batch["mask"])
    loss, metrics = policy_loss(output, batch["actions"], batch["mask"], 0.05, 0.01, 0.5)
    loss.backward()
    assert torch.isfinite(loss)
    assert -1.0 <= metrics["reward"] <= 1.05
    assert model.answer_head.weight.grad is not None
    assert model.vision.features[0].weight.grad is not None


def test_halting_is_bounded():
    chosen, log_probability = sample_halting(torch.zeros(32, 5))
    assert int(chosen.min()) >= 0
    assert int(chosen.max()) <= 4
    assert log_probability.shape == (32,)


def test_scale_configs_are_ordered():
    counts = []
    for scale in ("1m", "5m", "20m"):
        counts.append(parameter_count(BitterLessonAgent(**model_config(scale))))
    assert counts == sorted(counts)
    assert 900_000 <= counts[0] <= 1_200_000
    assert 4_500_000 <= counts[1] <= 5_500_000
    assert 18_000_000 <= counts[2] <= 22_000_000


def test_intermediate_scale_fills_the_optimization_ladder():
    small = parameter_count(BitterLessonAgent(**model_config("1m")))
    middle = parameter_count(BitterLessonAgent(**model_config("2m")))
    large = parameter_count(BitterLessonAgent(**model_config("5m")))
    assert small < middle < large


def test_latency_can_be_fully_disabled_during_sanity_runs():
    torch.manual_seed(4)
    batch = tiny_batch()
    model = BitterLessonAgent(hidden=48, memory_slots=4, heads=4, max_thought_steps=3)
    output = model(batch["frames"], batch["pcm"], batch["mask"])
    _, metrics = policy_loss(output, batch["actions"], batch["mask"],
                             speed_bonus=0.05, entropy_weight=0.0,
                             value_weight=0.5, latency_multiplier=0.0)
    assert metrics["latency_multiplier"] == 0.0
    assert metrics["reward"] in {-1.0, 0.0, 1.0}


def test_adaptive_curriculum_concentrates_on_frontier_with_replay():
    assert curriculum_sampling_choices((2,)) == (2,)
    assert curriculum_sampling_choices((2, 4)).count(2) == 3
    assert curriculum_sampling_choices((2, 4)).count(4) == 7
    choices = curriculum_sampling_choices((2, 4, 8))
    assert [choices.count(length) for length in (2, 4, 8)] == [3, 5, 12]
    choices = curriculum_sampling_choices((2, 4, 8, 16))
    assert [choices.count(length) for length in (2, 4, 8, 16)] == [2, 3, 5, 10]
    choices = curriculum_sampling_choices((2, 4, 8, 16, 32))
    assert [choices.count(length) for length in (2, 4, 8, 16, 32)] == [2, 3, 4, 5, 6]
    choices = curriculum_sampling_choices((2, 4, 8, 16, 32, 64))
    assert [choices.count(length) for length in (2, 4, 8, 16, 32, 64)] == [1, 2, 3, 4, 4, 6]
    choices = curriculum_sampling_choices((2, 4, 8, 16, 32, 64, 96))
    assert [choices.count(length) for length in (2, 4, 8, 16, 32, 64, 96)] == [1, 2, 3, 3, 3, 4, 6]


def test_fixed_thought_policy_does_not_train_halting():
    batch = tiny_batch()
    model = BitterLessonAgent(hidden=48, memory_slots=4, heads=4, max_thought_steps=2,
                              memory_core="event_transformer")
    output = model(batch["frames"], batch["pcm"], batch["mask"])
    loss, metrics = policy_loss(output, batch["actions"], batch["mask"],
                                speed_bonus=0.0, entropy_weight=0.0,
                                value_weight=0.5, fixed_thoughts=True)
    loss.backward()
    assert metrics["thought_steps"] == 2.0
    assert model.halt_head.weight.grad is None


def test_q_learning_uses_only_sampled_outcome_and_backpropagates():
    torch.manual_seed(5)
    batch = tiny_batch()
    model = BitterLessonAgent(hidden=48, memory_slots=4, heads=4, max_thought_steps=3)
    output = model(batch["frames"], batch["pcm"], batch["mask"])
    loss, metrics = q_learning_loss(output, batch["actions"], batch["mask"], epsilon=1.0)
    loss.backward()
    assert torch.isfinite(loss)
    assert metrics["exploration"] == 1.0
    assert model.answer_head.weight.grad is not None
    assert model.vision.features[0].weight.grad is not None


def test_verifier_control_has_no_semantic_targets():
    batch = tiny_batch()
    model = BitterLessonAgent(hidden=48, memory_slots=4, heads=4, max_thought_steps=3)
    output = model(batch["frames"], batch["pcm"], batch["mask"])
    loss, _ = verifier_loss(output, batch["actions"], batch["mask"])
    loss.backward()
    assert model.answer_head.weight.grad is not None
    assert model.vision.features[0].weight.grad is not None


def test_random_depth_consistency_backpropagates_without_semantic_targets():
    batch = tiny_batch()
    model = BitterLessonAgent(hidden=48, memory_slots=4, heads=4,
                              max_thought_steps=4, memory_core="event_transformer",
                              thought_dynamics="gated_residual")
    output = model(batch["frames"], batch["pcm"], batch["mask"])
    loss, metrics = randomized_depth_verifier_loss(
        output, batch["actions"], batch["mask"], consistency_weight=0.25)
    loss.backward()
    assert torch.isfinite(loss)
    assert metrics["consistency_loss"] >= -1e-6
    assert model.thought_gate.weight.grad is not None


def test_direct_sensory_control_bypasses_reasoning_head():
    batch = tiny_batch()
    model = BitterLessonAgent(hidden=48, memory_slots=4, heads=4, max_thought_steps=3)
    output = model(batch["frames"], batch["pcm"], batch["mask"])
    loss, _ = direct_sensory_loss(output, batch["actions"], batch["mask"])
    loss.backward()
    assert model.observation_head.weight.grad is not None
    assert model.answer_head.weight.grad is None
