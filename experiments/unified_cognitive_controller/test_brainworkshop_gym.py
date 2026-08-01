import torch

from .brainworkshop_gym import (
    ALL_MATCHES,
    AUDIO_MATCH,
    POSITION_MATCH,
    BrainWorkshopConfig,
    BrainWorkshopEventEncoders,
    BrainWorkshopGym,
    BrainWorkshopKeypressDecoder,
    generate_brainworkshop_episode,
)


def test_episode_is_seed_deterministic_and_targets_stay_private() -> None:
    config = BrainWorkshopConfig(n_back=2, trials=10)
    first = generate_brainworkshop_episode(config, seed=44001)
    duplicate = generate_brainworkshop_episode(config, seed=44001)
    assert first.seed == duplicate.seed
    assert first.stimuli == duplicate.stimuli
    assert first.verifier_targets() == duplicate.verifier_targets()
    assert all(not hasattr(observation, "target")
               for observation in first.observations)
    assert all(observation.vision is not None
               and observation.audio is not None
               for observation in first.observations)


def test_verifier_rewards_correctness_and_small_latency_bonus() -> None:
    episode = generate_brainworkshop_episode(
        BrainWorkshopConfig(n_back=1, trials=8), seed=44002)
    target = episode.verifier_targets()[3]
    fast = episode.score_action(3, target, latency_ms=0)
    slow = episode.score_action(3, target, latency_ms=1_000)
    wrong = episode.score_action(3, target ^ POSITION_MATCH, latency_ms=0)
    assert 1.0 < fast <= 1.05
    assert slow == 1.0
    assert fast > slow > wrong


def test_gym_returns_only_scalar_outcome_and_advances_real_time_trials() -> None:
    gym = BrainWorkshopGym(
        BrainWorkshopConfig(n_back=1, trials=4), seed=44003)
    observation = gym.reset()
    assert observation.timestamp_ms == 0
    for trial in range(4):
        observation, reward, done, info = gym.step(0, latency_ms=12.5)
        assert isinstance(reward, float)
        assert set(info) == {"latency_ms"}
        if trial < 3:
            assert observation is not None
            assert observation.timestamp_ms == (trial + 1) * 1_000
        else:
            assert done and observation is None


def test_both_streams_encode_as_independent_events() -> None:
    episode = generate_brainworkshop_episode(
        BrainWorkshopConfig(n_back=1, trials=4), seed=44004)
    encoders = BrainWorkshopEventEncoders(event_width=16)
    collection = encoders.encode(episode.observations[0])
    assert collection.payload.shape == (1, 2, 16)
    assert collection.present.tolist() == [[True, True]]
    assert collection.timestamp is not None
    assert collection.timestamp.tolist() == [[0.0, 0.0]]


def test_each_modality_can_be_connected_without_resizing_the_bus() -> None:
    for modalities in (("vision",), ("audio",)):
        episode = generate_brainworkshop_episode(
            BrainWorkshopConfig(n_back=1, trials=4, modalities=modalities),
            seed=44005,
        )
        collection = BrainWorkshopEventEncoders(event_width=16).encode(
            episode.observations[0])
        assert collection.payload.shape == (1, 1, 16)


def test_keypress_decoder_uses_two_bit_protocol() -> None:
    assert BrainWorkshopKeypressDecoder.to_keypress_codes(0) == ()
    assert BrainWorkshopKeypressDecoder.to_keypress_codes(POSITION_MATCH) == (1,)
    assert BrainWorkshopKeypressDecoder.to_keypress_codes(AUDIO_MATCH) == (2,)
    assert BrainWorkshopKeypressDecoder.to_keypress_codes(ALL_MATCHES) == (1, 2)
    decoder = BrainWorkshopKeypressDecoder(intention_width=6)
    assert decoder(torch.zeros(5, 6)).shape == (5, 4)
