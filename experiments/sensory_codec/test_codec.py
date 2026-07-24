from __future__ import annotations

import unittest

import numpy as np
import torch

from .model import DirectModel, Listener
from .games import GAME_NAMES, make_game, make_multigame_dataset
from .runtime import ACTION_TOKENS, SensoryPacket, VisualActionAgent, parse_action
from .snake import OPPOSITE, SnakeEnv, make_dataset
from .train import latency_audit, select_device
from .traps import modality_trap_audit


class SnakeTests(unittest.TestCase):
    def test_device_selection_has_cpu_fallback(self) -> None:
        self.assertEqual(select_device("cpu"), torch.device("cpu"))

    def test_observation_never_contains_privileged_state(self) -> None:
        env = SnakeEnv(seed=3)
        hidden = env.observe(apple_visible=False, tail_visible=False)
        self.assertEqual(float(hidden[3].sum()), 0.0)
        self.assertLess(float(hidden[1].sum()), len(env.state.snake))

    def test_teacher_is_legal_when_a_legal_move_exists(self) -> None:
        env = SnakeEnv(seed=8)
        for _ in range(100):
            legal = [action for action in range(4) if not env.danger(action)]
            action = env.teacher_action()
            if legal:
                self.assertIn(action, legal)
            _, done = env.step(action)
            if done:
                env.reset()

    def test_reverse_is_always_illegal(self) -> None:
        env = SnakeEnv(seed=4)
        self.assertTrue(env.danger(OPPOSITE[env.state.direction]))

    def test_dataset_shapes_and_reproducibility(self) -> None:
        first = make_dataset(12, 4, 8, 12)
        second = make_dataset(12, 4, 8, 12)
        self.assertEqual(first["frames"].shape, (12, 4, 4, 10, 10))
        np.testing.assert_array_equal(first["frames"], second["frames"])
        self.assertEqual(first["semantic"].shape, (12, 16))

    def test_model_contract(self) -> None:
        model = DirectModel(board_pixels=10)
        outputs, code, gate = model(torch.zeros(3, 4, 4, 10, 10))
        self.assertEqual(outputs["action"].shape, (3, 4))
        self.assertEqual(outputs["danger"].shape, (3, 4))
        self.assertEqual(code.shape, (3, 16))
        self.assertEqual(gate.shape, (3, 16))

    def test_listener_accepts_privileged_vector(self) -> None:
        outputs = Listener()(torch.zeros(2, 16))
        self.assertEqual(outputs["horizontal"].shape, (2, 3))

    def test_primitive_game_contracts(self) -> None:
        for name in GAME_NAMES:
            with self.subTest(game=name):
                env = make_game(name, size=8, seed=9)
                image = env.observe(target_visible=True, detail_visible=True)
                self.assertEqual(image.shape, (4, 10, 10))
                self.assertEqual(env.semantic_vector().shape, (16,))
                labels = env.labels()
                self.assertIn(labels["action"], range(4))
                self.assertEqual(labels["danger"].shape, (4,))
                reward, done = env.step(env.teacher_action())
                self.assertIsInstance(done, bool)
                self.assertTrue(np.isfinite(reward))

    def test_multigame_dataset_is_balanced(self) -> None:
        data = make_multigame_dataset(40, 4, 8, 21, GAME_NAMES)
        self.assertEqual(data["frames"].shape, (40, 4, 4, 10, 10))
        self.assertEqual(data["audio"].shape, (40, 4, 64))
        self.assertEqual(data["text"].shape, (40, 4, 32))
        counts = np.bincount(data["game"], minlength=len(GAME_NAMES))
        self.assertTrue(np.all(counts > 0), counts)

    def test_signal_exposes_raw_audio_and_visible_characters(self) -> None:
        env = make_game("signal", size=8, seed=4)
        self.assertGreater(float(np.abs(env.raw_audio()).sum()), 0.0)
        visible = bytes(int(value) for value in env.raw_text() if value).decode("ascii")
        self.assertTrue(visible.startswith("GO "), visible)
        # The target itself is not leaked into the visual target channel.
        self.assertEqual(float(env.observe()[3].sum()), 0.0)

    def test_mission_teachers_have_valid_oracles(self) -> None:
        for name in ("maze", "keydoor", "memory", "signal", "rhythm"):
            with self.subTest(game=name):
                env = make_game(name, size=10, seed=5)
                events = 0
                for _ in range(100):
                    reward, done = env.step(env.teacher_action())
                    self.assertFalse(done, f"{name} teacher died")
                    events += int(reward >= 1.0)
                self.assertGreater(events, 0, f"{name} teacher never completed its task")

        patrol = make_game("patrol", size=10, seed=5)
        for _ in range(50):
            _, done = patrol.step(patrol.teacher_action())
            self.assertFalse(done, "patrol teacher died")

    def test_latency_score_is_small_and_accuracy_dominant(self) -> None:
        data = make_multigame_dataset(4, 3, 8, 2, ("signal",))
        result = latency_audit(DirectModel(10), data, torch.device("cpu"),
                               runs=3, target_ms=50.0, latency_weight=0.002,
                               action_accuracy=0.8)
        self.assertGreater(result["mean_ms"], 0.0)
        self.assertLess(result["latency_penalty"], 0.05)
        self.assertLess(result["accuracy_dominant_score"], 0.8)
        self.assertGreater(result["accuracy_dominant_score"], 0.75)

    def test_modality_traps_are_complete_and_finite(self) -> None:
        result = modality_trap_audit(DirectModel(10), torch.device("cpu"), sequence=3, size=8)
        self.assertEqual(set(result["accuracy"]), {
            "congruent", "audio_only", "text_only", "corrupt_text",
            "corrupt_audio", "stale_audio", "stale_text",
        })
        self.assertTrue(np.isfinite(result["missing_entropy_delta"]))
        self.assertGreaterEqual(result["audio_minimal_pair_code_distance"], 0.0)

    def test_runtime_firewall_is_pixels_in_action_token_out(self) -> None:
        model = DirectModel(board_pixels=10)
        agent = VisualActionAgent(model, torch.device("cpu"))
        visual_history = np.zeros((4, 4, 10, 10), dtype=np.float32)
        packet = SensoryPacket.vision_only(visual_history)
        self.assertFalse(hasattr(packet, "game_state"))
        token = agent.emit(packet)
        self.assertIn(token, ACTION_TOKENS)
        self.assertIn(parse_action(token), range(4))
        with self.assertRaises(TypeError):
            agent.emit({"packet": packet, "game_state": "leak"})  # type: ignore[arg-type]
        with self.assertRaises(ValueError):
            agent.emit(SensoryPacket.vision_only(np.zeros((4, 5, 10, 10), dtype=np.float32)))


if __name__ == "__main__":
    unittest.main()
