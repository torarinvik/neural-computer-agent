from __future__ import annotations

import torch

from .external_caption_source import (
    ANNOTATION_TABLE_V3_PATH,
    corpus_payload,
    corpus_sha256,
    render_external_annotation_text_v3,
    render_external_text,
    render_external_text_v2,
)
from .train import (
    TEXT_LENGTH,
    ByteTextFrontend,
    ByteTransformerFrontend,
    render_grounded_text,
)


def _frames() -> torch.Tensor:
    frames = torch.zeros(4, 3, 8, 8)
    frames[0, 0, 1:4, 1:3] = 1.0
    frames[1, 1, 4:7, 5:7] = 1.0
    frames[2, 2, 1:4, 5:7] = 0.8
    frames[3, 0, 4:7, 1:3] = 0.8
    return frames


def test_byte_text_is_raw_padded_transport() -> None:
    frames = _frames()
    first = render_grounded_text(frames, style=0)
    second = render_grounded_text(frames, style=3)
    assert first.shape == (4, TEXT_LENGTH)
    assert first.dtype == torch.long
    assert int(first.min()) >= 0
    assert int(first.max()) <= 256
    assert torch.all(first[:, -1] == 0)
    assert not torch.equal(first, second)


def test_byte_frontend_emits_only_fixed_width_events() -> None:
    frontend = ByteTextFrontend(event_width=12)
    tokens = render_grounded_text(_frames(), style=[0, 1, 2, 4])
    output = frontend(tokens)
    assert output.shape == (4, 12)
    assert torch.isfinite(output).all()


def test_relative_order_frontend_emits_the_same_fixed_width_event() -> None:
    frontend = ByteTextFrontend(event_width=12, position_bins=4)
    tokens = render_grounded_text(_frames(), style=3)

    output = frontend(tokens)

    assert output.shape == (4, 12)
    assert torch.isfinite(output).all()


def test_byte_transformer_frontend_emits_fixed_width_events() -> None:
    frontend = ByteTransformerFrontend(event_width=12)
    tokens = render_grounded_text(_frames(), style=4)

    output = frontend(tokens)

    assert output.shape == (4, 12)
    assert torch.isfinite(output).all()


def test_paraphrase_views_describe_the_same_rendered_scene() -> None:
    frames = _frames()
    first = render_grounded_text(frames, style=0)
    second = render_grounded_text(frames, style=1)

    assert not torch.equal(first, second)
    assert first.shape == second.shape


def test_external_caption_corpus_is_pixel_only_and_versioned() -> None:
    frames = _frames()
    first = render_external_text(frames, style=0)
    second = render_external_text(frames, style=4)

    assert first.shape == (4, TEXT_LENGTH)
    assert second.shape == first.shape
    assert first.dtype == torch.long
    assert not torch.equal(first, second)
    assert len(corpus_sha256()) == 64
    assert torch.equal(first, render_external_text(frames, style=0))


def test_external_caption_v2_adds_training_order_variants() -> None:
    frames = _frames()
    first = render_external_text_v2(frames, style=0)
    second = render_external_text_v2(frames, style=0)

    # The source is deterministic for a given frame batch; variant selection
    # changes across positions, not through hidden sample metadata.
    assert torch.equal(first, second)
    assert first.shape == (4, TEXT_LENGTH)


def test_static_annotation_table_has_complete_captions_without_slots() -> None:
    frames = _frames()
    first = render_external_annotation_text_v3(frames, style=0)
    second = render_external_annotation_text_v3(frames, style=3)

    assert first.shape == (4, TEXT_LENGTH)
    assert second.shape == first.shape
    assert first.dtype == torch.long
    assert not torch.equal(first, second)
    assert len(corpus_sha256(ANNOTATION_TABLE_V3_PATH)) == 64
    assert torch.equal(first, render_external_annotation_text_v3(frames, style=0))
    payload = corpus_payload(ANNOTATION_TABLE_V3_PATH)
    assert payload["provenance"] == "static_full_sentence_annotations_without_runtime_slots"
    assert len(payload["records"]) == 24
    assert all(
        "{" not in sentence and "}" not in sentence
        for record in payload["records"]
        for sentence in (*record["train"], *record["heldout"])
    )
