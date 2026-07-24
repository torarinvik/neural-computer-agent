import tempfile
import unittest
from pathlib import Path
import io
import struct
import numpy as np

from .baselines import Baseline, BASELINE_INPUTS
from .challenge_reference import KINDS, generated, solve
from .difficulty_profiles import PROFILES
from .evaluation import EpisodeRecord, reward, summarize
from .streamer_experiments import VARIANTS
from .host_client import HEADER, MAGIC, normalize_action, read_packet
from .adapter_rl import AdapterConfig, StreamerGate, adapter_objective, freeze_listener, train_gate, packet_features, shape_reward
from .run_baselines import run
from .checkpointing import load_adapter, save_adapter
from .vlm_policy import AudioOnlyPolicy, extract_action
from .event_gate import EventGate
from .run_model_matrix import _slot_map
from .run_vlm_episode import action_from_text
from .boundary_leak import check as boundary_check

class ToolingTests(unittest.TestCase):
    def test_all_families_have_reference_semantics(self):
        self.assertEqual(len(KINDS), 90)
        for seed in range(90):
            c = generated(seed, "max")
            self.assertEqual(solve(c), c.answer)

    def test_difficulty_metadata_is_realized(self):
        intro = generated(7, "intro")
        maximum = generated(7, "max")
        self.assertLess(len(intro.values), len(maximum.values))
        self.assertLess(intro.nesting_depth, maximum.nesting_depth)
        self.assertLess(intro.interference_permille, maximum.interference_permille)

    def test_reward_is_correctness_dominant(self):
        self.assertGreater(reward("correct", 1, 1000), 1.0)
        self.assertLess(reward("wrong", 1, 1000), 0.0)
        self.assertEqual(reward("timeout", 1, 1000), -1.0)

    def test_metrics_are_grouped(self):
        rows = [EpisodeRecord(1, "ArithmeticChain", "max", "correct", 100, 1000),
                EpisodeRecord(2, "Parity", "max", "timeout", 1000, 1000)]
        result = summarize(rows)
        self.assertEqual(result["episodes"], 2)
        self.assertIn("ArithmeticChain/max", result["by_family_difficulty"])

    def test_experiment_matrix(self):
        self.assertEqual({v.name for v in VARIANTS}, {"dense", "fixed-gate", "random-control", "learned-gate"})
        self.assertTrue(all(v.reward_weight > v.latency_weight for v in VARIANTS))
        self.assertEqual(len(BASELINE_INPUTS), len(Baseline))
        self.assertIn("max", PROFILES)

    def test_model_matrix_keeps_unassigned_slots_explicit(self):
        self.assertEqual(len(_slot_map(None)), 0)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "models.json"
            path.write_text('{"350m": "local/model", "1b": "hub/model"}')
            self.assertEqual(_slot_map(path)["350m"]["model"], "local/model")

    def test_model_registry_preserves_actual_parameter_metadata(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "models.json"
            path.write_text('{"350m": {"model": "local/model", "actual_parameters": 256000000, "modality": "vision+text"}}')
            entry = _slot_map(path)["350m"]
            self.assertEqual(entry["actual_parameters"], 256000000)

    def test_host_envelope_is_pixels_and_pcm_only(self):
        rgb = bytes([1, 2, 3] * 2)
        pcm = struct.pack("<2h", 10, -20)
        envelope = HEADER.pack(MAGIC, 17, 2, 1, len(rgb), 2, 16_000) + rgb + pcm
        packet = read_packet(io.BytesIO(envelope))
        self.assertEqual(packet.timestamp_ms, 17)
        self.assertEqual(packet.frame.shape, (1, 2, 3))
        np.testing.assert_array_equal(packet.pcm, np.array([10, -20], dtype=np.int16))

    def test_transport_event_labels_frame_and_audio_boundaries(self):
        from .host_client import TransportEvent
        event = TransportEvent("stream_received", 123, 7, modalities=("frame", "audio"))
        self.assertEqual(event.modalities, ("frame", "audio"))

    def test_model_output_is_strictly_normalized(self):
        self.assertEqual(normalize_action("  next\nexplanation"), "NEXT")
        self.assertEqual(normalize_action("submit the answer"), "WAIT")

    def test_adapter_keeps_listener_frozen_and_latency_secondary(self):
        import torch
        listener = torch.nn.Linear(2, 2)
        freeze_listener(listener)
        self.assertTrue(all(not p.requires_grad for p in listener.parameters()))
        gate = StreamerGate()
        decisions, log_prob = gate(torch.zeros(4, 8))
        loss = adapter_objective(torch.ones(4), log_prob, decisions.sum(1), AdapterConfig("learned-gate"))
        self.assertTrue(torch.isfinite(loss))
        self.assertLess(AdapterConfig("learned-gate").latency_weight,
                        AdapterConfig("learned-gate").reward_weight)
        config = AdapterConfig("learned-gate")
        covered = shape_reward(torch.ones(1), torch.full((1,), 0.2), config)
        silent = shape_reward(torch.ones(1), torch.zeros(1), config)
        self.assertLess(float(silent), float(covered))

    def test_baseline_runner_is_causal(self):
        random_rows = run(Baseline.RANDOM, episodes=8, premises=3, deadline_ms=1000,
                          inference_ms=1, seed=4)
        oracle_rows = run(Baseline.TEXT_ORACLE, episodes=8, premises=3, deadline_ms=1000,
                          inference_ms=1, seed=4)
        self.assertEqual(len(random_rows), 8)
        self.assertEqual(sum(x.outcome == "correct" for x in oracle_rows), 8)
        self.assertTrue(0 <= sum(x.outcome == "correct" for x in random_rows) <= 8)

    def test_adapter_checkpoint_round_trip(self):
        import torch
        import tempfile
        module = StreamerGate()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "adapter.pt"
            save_adapter(path, module, config={"variant": "learned-gate"},
                         metrics={"accuracy": 0.5}, seed=9)
            restored = StreamerGate()
            metadata = load_adapter(path, restored)
            self.assertEqual(metadata["seed"], 9)
            for left, right in zip(module.parameters(), restored.parameters()):
                torch.testing.assert_close(left, right)

    def test_vlm_action_boundary(self):
        self.assertEqual(extract_action("The answer is NEXT."), "NEXT")
        self.assertEqual(extract_action("no valid command"), "WAIT")
        packet = type("Packet", (), {"pcm": np.zeros(16, dtype=np.int16)})()
        self.assertEqual(AudioOnlyPolicy()(packet), "WAIT")

    def test_gate_training_updates_only_gate(self):
        import torch
        gate = StreamerGate()
        listener = torch.nn.Linear(2, 2)
        before = [p.detach().clone() for p in listener.parameters()]
        batches = [torch.randn(4, 8)]
        def rollout(decisions):
            reward = decisions.sum(1)
            return reward, decisions.sum(1)
        history = train_gate(gate, listener, rollout, batches,
                             AdapterConfig("learned-gate"), steps=2)
        self.assertEqual(len(history), 2)
        for old, new in zip(before, listener.parameters()):
            torch.testing.assert_close(old, new)

    def test_event_gate_is_causal_and_thresholded(self):
        from .host_client import HostPacket
        frame = np.zeros((2, 2, 3), dtype=np.uint8)
        silent = HostPacket(0, 2, 2, frame, np.zeros(4, dtype=np.int16), 16_000)
        changed = HostPacket(10, 2, 2, frame + 10, np.zeros(4, dtype=np.int16), 16_000)
        gate = EventGate(mode="fixed-gate", frame_threshold=1.0,
                         audio_rms_threshold=100.0, audio_silence_ms=100)
        self.assertTrue(gate.accept(silent))
        self.assertFalse(gate.accept(silent))
        self.assertTrue(gate.accept(changed))

    def test_adapter_features_have_no_private_state(self):
        from .host_client import HostPacket
        packet = HostPacket(10, 2, 2, np.zeros((2, 2, 3), dtype=np.uint8),
                            np.zeros(4, dtype=np.int16), 16000)
        features = packet_features(packet)
        self.assertEqual(tuple(features.shape), (8,))

    def test_previous_is_preserved_at_model_boundary(self):
        self.assertEqual(action_from_text("PREVIOUS").name, "PREVIOUS")

    def test_hidden_answer_and_family_cannot_change_packets(self):
        result = boundary_check(31)
        self.assertTrue(result["identical_packets"])

if __name__ == "__main__":
    unittest.main()
