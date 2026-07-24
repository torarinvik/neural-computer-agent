from __future__ import annotations

import unittest

import numpy as np

from .environment import MODALITIES, make_dataset


class ContinuousReflexTests(unittest.TestCase):
    def test_balanced_labels(self):
        data = make_dataset(1200, 32, 10, 4)
        actions = np.bincount(data["action"])
        modalities = np.bincount(data["modality"], minlength=len(MODALITIES))
        self.assertLessEqual(actions.max() - actions.min(), 1)
        self.assertLessEqual(modalities.max() - modalities.min(), 1)

    def test_only_raw_streams_enter_model_contract(self):
        data = make_dataset(12, 24, 8, 5)
        self.assertEqual(data["frames"].shape, (12, 24, 4, 10, 10))
        self.assertEqual(data["audio"].shape, (12, 24, 64))
        self.assertEqual(data["relevant_ticks"].shape, (12, 24, 2))

    def test_each_trial_has_relevant_event(self):
        data = make_dataset(120, 32, 10, 6)
        expected = np.where(data["modality"] == 2, 2, 1)
        np.testing.assert_array_equal(data["relevant_ticks"].sum((1, 2)), expected)

    def test_distractors_outnumber_relevant_events(self):
        data = make_dataset(40, 32, 10, 7, distractors=7)
        visual_changes = np.abs(np.diff(data["frames"], axis=1)).sum((2, 3, 4)) > 0
        audio_events = np.abs(data["audio"]).sum(-1) > 0
        total_events = visual_changes.sum(1) + audio_events.sum(1)
        self.assertTrue((total_events > data["relevant_ticks"].sum((1, 2))).all())

    def test_heldout_sensors_shift_appearance(self):
        train = make_dataset(24, 32, 10, 8, "train")
        heldout = make_dataset(24, 32, 10, 8, "heldout")
        self.assertFalse(np.array_equal(train["frames"], heldout["frames"]))
        self.assertFalse(np.array_equal(train["audio"], heldout["audio"]))


if __name__ == "__main__":
    unittest.main()
