import torch

from .train_amodal_text_alignment import (
    PairRelationTextEncoder,
    render_pair_relation_text_tokens,
)


def test_text_sensor_is_a_fixed_discrete_serialization() -> None:
    frames = torch.rand(5, 3, 32, 32)
    tokens = render_pair_relation_text_tokens(frames)
    assert tokens.shape == (5, 16 * 16 * 3)
    assert tokens.dtype == torch.long
    assert int(tokens.min()) >= 0 and int(tokens.max()) < 16
    assert not torch.equal(tokens, render_pair_relation_text_tokens(frames.roll(1, -1)))


def test_text_encoder_lowers_symbols_to_opaque_event_width() -> None:
    tokens = render_pair_relation_text_tokens(torch.rand(7, 3, 32, 32))
    events = PairRelationTextEncoder(96)(tokens)
    assert events.shape == (7, 96)
    assert torch.isfinite(events).all()
