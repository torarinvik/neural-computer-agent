from __future__ import annotations

import unittest

import numpy as np

from .environment import MODALITIES, directional_pcm, make_reflex_dataset, render_frame


class ReflexTests(unittest.TestCase):
    def test_actions_are_balanced(self) -> None:
        data = make_reflex_dataset(1200, 12, 10, 3)
        counts = np.bincount(data["action"], minlength=4)
        self.assertLessEqual(int(counts.max() - counts.min()), 1)

    def test_modalities_are_balanced(self) -> None:
        data = make_reflex_dataset(1200, 12, 10, 4)
        counts = np.bincount(data["modality"], minlength=len(MODALITIES))
        self.assertLessEqual(int(counts.max() - counts.min()), 1)

    def test_audio_trials_contain_one_event_and_silence(self) -> None:
        data = make_reflex_dataset(60, 12, 10, 5)
        audio_only = data["audio"][data["modality"] == 1]
        active = np.abs(audio_only).sum(-1) > 0
        self.assertTrue((active.sum(-1) == 1).all())

    def test_directional_pcm_is_raw_and_distinct(self) -> None:
        waves = [directional_pcm(direction, False) for direction in range(4)]
        self.assertTrue(all(wave.shape == (64,) for wave in waves))
        self.assertTrue(all(not np.array_equal(waves[0], wave) for wave in waves[1:]))

    def test_visual_cue_crosses_fixed_event_threshold(self) -> None:
        baseline = render_frame(10, None, False)
        cue = render_frame(10, 0, False)
        self.assertGreater(np.abs(cue - baseline).mean(), 0.002)

    def test_dataset_exposes_only_raw_sensors_and_labels(self) -> None:
        data = make_reflex_dataset(8, 8, 8, 6)
        self.assertEqual(set(data), {"frames", "audio", "action", "modality", "hazard"})
        self.assertEqual(data["frames"].shape, (8, 8, 4, 10, 10))
        self.assertEqual(data["audio"].shape, (8, 8, 64))


if __name__ == "__main__":
    unittest.main()
