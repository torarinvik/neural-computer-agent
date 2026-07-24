from __future__ import annotations

import unittest

import numpy as np

from .environment import Action, RealtimeEpisode, generate_question


class FakeClock:
    def __init__(self):
        self.ns = 0

    def __call__(self):
        return self.ns

    def advance_ms(self, value):
        self.ns += value * 1_000_000


class RealtimeSyllogimousTests(unittest.TestCase):
    def test_generation_is_deterministic_and_balanced(self):
        self.assertEqual(generate_question(7), generate_question(7))
        answers = [generate_question(seed).answer for seed in range(400)]
        self.assertGreater(sum(answers), 160)
        self.assertLess(sum(answers), 240)

    def test_reserved_seeds_are_protected(self):
        with self.assertRaises(ValueError):
            generate_question(100_000)
        self.assertEqual(generate_question(100_000, final=True).seed, 100_000)

    def test_observation_contains_only_pixels_pcm_and_timestamp(self):
        clock = FakeClock()
        result = RealtimeEpisode(generate_question(4), clock_ns=clock).step(Action.WAIT)
        self.assertEqual(result.observation.frame.shape, (400, 640, 3))
        self.assertEqual(result.observation.frame.dtype, np.uint8)
        self.assertEqual(result.observation.pcm.shape, (533,))
        self.assertEqual(result.observation.pcm.dtype, np.float32)
        self.assertFalse(hasattr(result.observation, "answer"))
        self.assertFalse(hasattr(result.observation, "premises"))

    def test_answer_is_rejected_before_conclusion(self):
        episode = RealtimeEpisode(generate_question(8), clock_ns=FakeClock())
        result = episode.step(Action.TRUE)
        self.assertFalse(result.done)
        self.assertEqual(result.reward, 0.0)

    def test_previous_action_backtracks_causally(self):
        clock = FakeClock()
        episode = RealtimeEpisode(generate_question(12, premises=3), clock_ns=clock)
        episode.step(Action.NEXT)
        result = episode.step(Action.PREVIOUS)
        self.assertFalse(result.done)
        self.assertEqual(result.observation.frame.shape, (400, 640, 3))
        self.assertEqual(result.observation.pcm.dtype, np.float32)

    def test_correct_speed_bonus_is_small_and_wrong_never_gets_it(self):
        clock = FakeClock()
        question = generate_question(9, premises=2)
        episode = RealtimeEpisode(question, deadline_ms=1000, clock_ns=clock)
        episode.step(Action.NEXT)
        episode.step(Action.NEXT)
        clock.advance_ms(500)
        correct = Action.TRUE if question.answer else Action.FALSE
        result = episode.step(correct)
        self.assertAlmostEqual(result.reward, 1.025)
        clock2 = FakeClock()
        episode2 = RealtimeEpisode(question, deadline_ms=1000, clock_ns=clock2)
        episode2.step(Action.NEXT)
        episode2.step(Action.NEXT)
        result2 = episode2.step(Action.FALSE if question.answer else Action.TRUE)
        self.assertEqual(result2.reward, -1.0)

    def test_clock_runs_during_inference_and_late_answer_times_out(self):
        clock = FakeClock()
        question = generate_question(10, premises=2)
        episode = RealtimeEpisode(question, deadline_ms=100, clock_ns=clock)
        episode.step(Action.NEXT)
        episode.step(Action.NEXT)
        clock.advance_ms(101)
        result = episode.step(Action.TRUE if question.answer else Action.FALSE)
        self.assertTrue(result.done)
        self.assertEqual(result.outcome, "timeout")
        self.assertEqual(result.reward, -1.0)


if __name__ == "__main__":
    unittest.main()
