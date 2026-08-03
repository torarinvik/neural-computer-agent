import torch

from .train_amodal_caption_alignment import (
    PairRelationCaptionEncoder,
    render_pair_relation_captions,
)


def test_caption_frontend_emits_bounded_word_ids() -> None:
    frames = torch.rand(6, 3, 32, 32)
    captions = render_pair_relation_captions(frames, torch.arange(6) % 2)
    assert captions.shape == (6, 8)
    assert captions.dtype == torch.long
    assert int(captions.min()) >= 0 and int(captions.max()) < 64


def test_caption_encoder_lowers_words_to_opaque_events() -> None:
    frames = torch.rand(5, 3, 32, 32)
    captions = render_pair_relation_captions(frames, torch.zeros(5, dtype=torch.long))
    events = PairRelationCaptionEncoder(96)(captions)
    assert events.shape == (5, 96)
    assert torch.isfinite(events).all()
