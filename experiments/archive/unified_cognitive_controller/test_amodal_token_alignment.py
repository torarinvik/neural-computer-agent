import torch

from .train_amodal_token_alignment import (
    PairRelationTokenEncoder,
    render_pair_relation_tokens,
)


def test_token_sensor_preserves_fixed_grid_and_is_semantics_free() -> None:
    frames = torch.rand(5, 3, 32, 32)
    tokens = render_pair_relation_tokens(frames)
    assert tokens.shape == (5, 16 * 16, 3)
    assert tokens.dtype.is_floating_point
    assert torch.all((tokens >= 0) & (tokens <= 1))
    shifted = render_pair_relation_tokens(frames.roll(1, dims=-1))
    assert not torch.equal(tokens, shifted)


def test_token_encoder_emits_only_the_opaque_event_width() -> None:
    tokens = render_pair_relation_tokens(torch.rand(7, 3, 32, 32))
    encoder = PairRelationTokenEncoder(96)
    events = encoder(tokens)
    assert events.shape == (7, 96)
    assert torch.isfinite(events).all()
