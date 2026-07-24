from __future__ import annotations

import inspect
import unittest

import numpy as np
import torch

from .data import (EpisodeDataset, balanced_question, collate_episodes,
                   generate_public_episode, visible_texts)
from .model import AgentOutput, LatentAgent, parameter_count
from .train import supervised_loss


class LatentAgentTests(unittest.TestCase):
    def test_public_episode_is_deterministic_and_raw(self):
        left = generate_public_episode(11, 3)
        right = generate_public_episode(11, 3)
        np.testing.assert_array_equal(left.frames, right.frames)
        np.testing.assert_array_equal(left.pcm, right.pcm)
        self.assertEqual(left.frames.dtype, np.uint8)
        self.assertEqual(left.pcm.dtype, np.float32)

    def test_model_boundary_has_no_private_state(self):
        parameters = tuple(inspect.signature(LatentAgent.forward).parameters)
        self.assertEqual(parameters, ("self", "frames", "pcm", "mask"))

    def test_collation_is_causal_and_padded(self):
        batch = collate_episodes([generate_public_episode(1, 2),
                                  generate_public_episode(2, 4)])
        self.assertEqual(tuple(batch["frames"].shape[:2]), (2, 5))
        self.assertEqual(batch["mask"].sum(1).tolist(), [3, 5])
        self.assertTrue((batch["actions"][0, 3:] == -100).all())

    def test_all_cores_emit_direct_action_logits(self):
        batch = collate_episodes([EpisodeDataset(1)[0]])
        for core in ("gru", "graph", "graph_cached", "closure", "recursive"):
            model = LatentAgent(core=core, hidden=96, recursive_steps=2)
            result = model(batch["frames"], batch["pcm"], batch["mask"])
            self.assertEqual(tuple(result.logits.shape), (1, 3, 5))
            self.assertEqual(tuple(result.halt_logits.shape), (1, 3))
            self.assertEqual(tuple(result.subject_logits.shape), (1, 3, 64))
            self.assertEqual(tuple(result.relation_logits.shape), (1, 3, 8))

    def test_default_models_are_in_small_model_range(self):
        for core in ("gru", "graph", "graph_cached", "closure", "recursive"):
            count = parameter_count(LatentAgent(core=core))
            self.assertGreater(count, 2_500_000)
            self.assertLess(count, 7_000_000)

    def test_heldout_symbols_reach_pixels_not_metadata(self):
        train = generate_public_episode(10, 2, heldout=False)
        heldout = generate_public_episode(10, 2, heldout=True)
        self.assertFalse(np.array_equal(train.frames, heldout.frames))
        np.testing.assert_array_equal(train.actions, heldout.actions)

    def test_final_answer_loss_cannot_be_hidden_by_next_actions(self):
        actions = torch.tensor([[1, 1, 3], [1, 1, 2]])
        mask = torch.ones_like(actions, dtype=torch.bool)
        logits = torch.full((2, 3, 5), -5.0)
        logits[:, :, 1] = 5.0
        output = AgentOutput(logits, torch.zeros(2, 3),
                             torch.zeros(2, 3, 64), torch.zeros(2, 3, 8),
                             torch.zeros(2, 3, 64), torch.zeros(2, 3))
        batch = {"actions": actions, "mask": mask,
                 "subjects": torch.zeros(2, 3, dtype=torch.long),
                 "relations": torch.zeros(2, 3, dtype=torch.long),
                 "objects": torch.zeros(2, 3, dtype=torch.long)}
        loss = supervised_loss(output, batch, torch.nn.CrossEntropyLoss())
        self.assertGreater(float(loss), 5.0)

    def test_every_conclusion_relation_occurs_with_both_answers(self):
        observed: dict[str, set[bool]] = {}
        for seed in range(2000):
            question = balanced_question(seed, 3)
            _, tail = question.conclusion.split(" IS ", 1)
            relation, _ = tail.rsplit(" ", 1)
            observed.setdefault(relation, set()).add(question.answer)
        self.assertEqual(len(observed), 8)
        self.assertTrue(all(answers == {False, True} for answers in observed.values()))

    def test_style_is_independent_of_seed_and_label(self):
        from .data import render_public_card
        left = render_public_card("Q01 IS BEFORE Q02", 1, 3, 0)
        right = render_public_card("Q01 IS BEFORE Q02", 1, 3, 0)
        np.testing.assert_array_equal(left, right)

    def test_conclusion_marker_does_not_encode_answer(self):
        from .data import render_public_card
        true_card = render_public_card("Q01 IS BEFORE Q02", 3, 3, 0, is_final=True)
        false_card = render_public_card("Q01 IS AFTER Q02", 3, 3, 0, is_final=True)
        self.assertEqual(true_card.shape, false_card.shape)
        premise = render_public_card("Q01 IS BEFORE Q02", 2, 3, 0, is_final=False)
        self.assertFalse(np.array_equal(true_card[:25], premise[:25]))

    def test_z_prefix_is_reserved_for_heldout_evaluation(self):
        train_prefixes = {
            visible_texts(balanced_question(seed, 2), heldout=False)[0][0]
            for seed in range(12)
        }
        self.assertNotIn("Z", train_prefixes)
        heldout = visible_texts(balanced_question(7, 2, heldout=True), heldout=True)
        self.assertTrue(all(text.startswith("Z") for text in heldout))

    def test_extended_generator_supports_64_premises(self):
        episode = generate_public_episode(100_064, 64, heldout=True, final=True,
                                          entity_count=128)
        self.assertEqual(episode.length, 65)
        self.assertLess(int(episode.subjects.max()), 128)
        self.assertLess(int(episode.objects.max()), 128)

    def test_cached_stream_matches_full_causal_forward(self):
        torch.manual_seed(3)
        batch = collate_episodes([generate_public_episode(4, 3)])
        model = LatentAgent(core="graph_cached", hidden=96, entity_count=64).eval()
        with torch.no_grad():
            full = model(batch["frames"], batch["pcm"], batch["mask"]).logits
            state = model.init_stream_state()
            streamed = []
            for index in range(batch["frames"].shape[1]):
                output, state = model.stream_step(batch["frames"][:, index],
                                                  batch["pcm"][:, index], state)
                streamed.append(output.logits)
        torch.testing.assert_close(full, torch.cat(streamed, dim=1), rtol=1e-5, atol=1e-5)

    def test_position_free_cache_matches_full_causal_forward(self):
        torch.manual_seed(5)
        batch = collate_episodes([generate_public_episode(6, 4)])
        model = LatentAgent(core="graph_cached", hidden=96,
                            entity_count=64, use_positions=False).eval()
        self.assertFalse(hasattr(model, "positions"))
        with torch.no_grad():
            full = model(batch["frames"], batch["pcm"], batch["mask"]).logits
            state = model.init_stream_state()
            streamed = []
            for index in range(batch["frames"].shape[1]):
                output, state = model.stream_step(batch["frames"][:, index],
                                                  batch["pcm"][:, index], state)
                streamed.append(output.logits)
        torch.testing.assert_close(full, torch.cat(streamed, dim=1), rtol=1e-5, atol=1e-5)

    def test_closure_answer_loss_does_not_corrupt_parser_gradients(self):
        batch = collate_episodes([generate_public_episode(8, 3)])
        model = LatentAgent(core="closure", hidden=96, entity_count=64)
        output = model(batch["frames"], batch["pcm"], batch["mask"])
        output.logits.sum().backward()
        self.assertIsNone(model.subject_head.weight.grad)
        self.assertIsNone(model.relation_head.weight.grad)
        self.assertIsNone(model.object_head.weight.grad)


if __name__ == "__main__":
    unittest.main()
