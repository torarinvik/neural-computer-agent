from __future__ import annotations

import unittest

import numpy as np
import torch

from .environment import event_audio, make_event_dataset
from .model import EventAdapter
from .runtime import AVPacket
from .train import objective


class EventStreamTests(unittest.TestCase):
    def test_audio_is_pcm_and_silence_is_exact(self) -> None:
        self.assertEqual(float(np.abs(event_audio(False)).sum()), 0.0)
        self.assertGreater(float(np.abs(event_audio(True)).sum()), 1.0)

    def test_dataset_contains_repeated_frames_and_silence(self) -> None:
        data = make_event_dataset(16, sequence=9, size=8, seed=3, sensor_ticks=3)
        deltas = np.abs(np.diff(data["frames"], axis=1)).sum((2, 3, 4))
        self.assertTrue((deltas == 0).any())
        self.assertTrue((data["audio"] == 0).all(-1).any())

    def test_fixed_adapter_suppresses_redundant_events(self) -> None:
        adapter = EventAdapter(10, 32, mode="fixed")
        frames = torch.zeros(1, 6, 4, 10, 10)
        audio = torch.zeros(1, 6, 64)
        _, _, audit = adapter(frames, audio)
        self.assertEqual(float(audit["vision_emissions"].item()), 1.0)
        self.assertEqual(float(audit["audio_emissions"].item()), 0.0)

    def test_runtime_packet_has_no_privileged_channel(self) -> None:
        packet = AVPacket(np.zeros((3, 4, 10, 10), np.float32),
                          np.zeros((3, 64), np.float32))
        packet.validate()
        self.assertEqual(set(packet.__slots__), {"vision", "audio"})

    def test_efficiency_cannot_dominate_accuracy(self) -> None:
        accurate = objective(torch.tensor(0.10), torch.tensor(20.0), 0.0005)
        inaccurate = objective(torch.tensor(0.20), torch.tensor(0.0), 0.0005)
        self.assertLess(float(accurate), float(inaccurate))


if __name__ == "__main__":
    unittest.main()
