"""Independent corpus-backed captions for the next text boundary.

The source accepts rendered pixels only.  Its phrasing corpus is separate from
the original template renderer and is loaded as versioned data.  No lifetime,
identity, context, action, or verifier metadata is accepted by this module.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch

TEXT_LENGTH = 128
TRAIN_STYLES = (0, 1, 2)
HELDOUT_STYLES = (3, 4)
ALL_STYLES = TRAIN_STYLES + HELDOUT_STYLES
CORPUS_PATH = Path(__file__).with_name("external_caption_corpus.json")
CORPUS_V2_PATH = Path(__file__).with_name("external_caption_corpus_v2.json")
ANNOTATION_TABLE_V3_PATH = Path(__file__).with_name(
    "external_caption_annotation_table_v3.json"
)


def corpus_payload(path: Path = CORPUS_PATH) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def corpus_sha256(path: Path = CORPUS_PATH) -> str:
    import hashlib

    return hashlib.sha256(path.read_bytes()).hexdigest()


def _pixel_facts(
    frames: torch.Tensor,
) -> tuple[list[str], list[str], list[str], list[str], list[str]]:
    """Describe visible pixels without accepting any renderer metadata."""
    if frames.ndim != 4 or frames.shape[1] != 3:
        raise ValueError("frames must have shape [batch, 3, height, width]")
    max_value = frames.max(dim=1).values
    min_value = frames.min(dim=1).values
    visible = (max_value - min_value > 0.12) & (max_value > 0.15)
    height, width = frames.shape[-2:]
    ys = torch.arange(height, device=frames.device).view(1, height, 1)
    xs = torch.arange(width, device=frames.device).view(1, 1, width)
    diagonal_visible = visible & (xs < ys)
    diagonal_mass = diagonal_visible.float().sum((1, 2))
    selected = torch.where(
        (diagonal_mass >= 8.0)[:, None, None], diagonal_visible, visible
    )
    mass = selected.float().sum((1, 2)).clamp_min(1.0)
    rgb = (frames * selected.unsqueeze(1)).sum((2, 3)) / mass.unsqueeze(1)
    colour_names = ("red", "green", "blue")
    colours = [colour_names[int(value)] for value in rgb.argmax(dim=-1)]
    brightness = [
        "bright" if bool(value) else "dim"
        for value in (rgb.mean(dim=-1) > 0.45)
    ]
    centre_y = (selected * ys).sum((1, 2)) / mass
    centre_x = (selected * xs).sum((1, 2)) / mass
    horizontal = ["left" if value <= width / 2 else "right" for value in centre_x]
    vertical = ["upper" if value <= height / 2 else "lower" for value in centre_y]
    variance_y = (
        selected * (ys - centre_y[:, None, None]).square()
    ).sum((1, 2)) / mass
    variance_x = (
        selected * (xs - centre_x[:, None, None]).square()
    ).sum((1, 2)) / mass
    shapes = [
        "tall form" if value else "wide form"
        for value in (variance_y.sqrt() >= variance_x.sqrt())
    ]
    return colours, brightness, horizontal, vertical, shapes


def _article(word: str) -> str:
    return "an" if word[0].lower() in "aeiou" else "a"


def _render_external_text(
    frames: torch.Tensor,
    *,
    style: int | list[int] = 0,
    text_length: int = TEXT_LENGTH,
    corpus_path: Path = CORPUS_PATH,
    train_variants: bool = False,
) -> torch.Tensor:
    """Encode corpus-backed captions generated from pixels only."""
    if text_length < 32:
        raise ValueError("text_length is too small for external captions")
    if isinstance(style, int):
        styles = [style] * frames.shape[0]
    else:
        styles = list(style)
        if len(styles) != frames.shape[0]:
            raise ValueError("style sequence must match the frame batch")
    if any(value not in ALL_STYLES for value in styles):
        raise ValueError("unknown external caption style")
    payload = corpus_payload(corpus_path)
    templates = payload["templates"]
    colours, brightness, horizontal, vertical, shapes = _pixel_facts(frames)
    sentences: list[str] = []
    for index, selected_style in enumerate(styles):
        split = "train" if selected_style in TRAIN_STYLES else "heldout"
        templates_for_split = templates[split]
        template_index = selected_style - len(TRAIN_STYLES) if split == "heldout" else selected_style
        if split == "train" and train_variants:
            template_index += (index % 2) * len(TRAIN_STYLES)
        template = templates_for_split[template_index]
        sentence = template.format(
            colour=colours[index],
            brightness=brightness[index],
            horizontal=horizontal[index],
            vertical=vertical[index],
            shape=shapes[index],
            article=_article(colours[index]),
        )
        encoded = sentence.encode("utf-8")
        if len(encoded) > text_length:
            raise ValueError("external caption exceeds text_length")
        sentences.append(sentence)
    result = torch.zeros(
        frames.shape[0], text_length, dtype=torch.long, device=frames.device
    )
    for index, sentence in enumerate(sentences):
        raw = torch.tensor(
            [value + 1 for value in sentence.encode("utf-8")],
            dtype=torch.long,
            device=frames.device,
        )
        result[index, : raw.numel()] = raw
    return result


def render_external_text(
    frames: torch.Tensor,
    *,
    style: int | list[int] = 0,
    text_length: int = TEXT_LENGTH,
) -> torch.Tensor:
    """Encode the original controlled external caption corpus."""
    return _render_external_text(
        frames, style=style, text_length=text_length, corpus_path=CORPUS_PATH
    )


def render_external_text_v2(
    frames: torch.Tensor,
    *,
    style: int | list[int] = 0,
    text_length: int = TEXT_LENGTH,
) -> torch.Tensor:
    """Encode v2 with two authored train variants per style."""
    return _render_external_text(
        frames,
        style=style,
        text_length=text_length,
        corpus_path=CORPUS_V2_PATH,
        train_variants=True,
    )


def _annotation_key(
    frames: torch.Tensor,
) -> list[str]:
    """Return static-table keys from visible pixels only.

    The key is used only to join an image to a pre-authored annotation row.
    No sentence is assembled from the extracted values; the complete caption
    is read from the versioned table below.
    """
    colours, _brightness, horizontal, vertical, shapes = _pixel_facts(frames)
    return [
        f"{colour}|{side}|{height}|{shape}"
        for colour, side, height, shape in zip(
            colours, horizontal, vertical, shapes, strict=True
        )
    ]


def render_external_annotation_text_v3(
    frames: torch.Tensor,
    *,
    style: int | list[int] = 0,
    text_length: int = TEXT_LENGTH,
) -> torch.Tensor:
    """Encode complete pre-authored captions joined to pixels by annotation.

    This is deliberately a separate data protocol from the v1/v2 phrase
    renderers: the corpus contains full sentences and no runtime format slots.
    The source still sees pixels to perform the ordinary image/annotation join,
    but it never receives verifier metadata or synthesizes text from it.
    """
    if text_length < 32:
        raise ValueError("text_length is too small for external annotations")
    if isinstance(style, int):
        styles = [style] * frames.shape[0]
    else:
        styles = list(style)
        if len(styles) != frames.shape[0]:
            raise ValueError("style sequence must match the frame batch")
    if any(value not in ALL_STYLES for value in styles):
        raise ValueError("unknown external annotation style")
    payload = corpus_payload(ANNOTATION_TABLE_V3_PATH)
    records = {str(row["key"]): row for row in payload["records"]}
    if len(records) != len(payload["records"]):
        raise ValueError("annotation table contains duplicate keys")
    keys = _annotation_key(frames)
    sentences: list[str] = []
    for key, selected_style in zip(keys, styles, strict=True):
        row = records.get(key)
        if row is None:
            raise ValueError(f"annotation table has no row for visible key {key!r}")
        split = "train" if selected_style in TRAIN_STYLES else "heldout"
        variants = row[split]
        sentence = str(variants[selected_style if split == "train" else selected_style - 3])
        if "{" in sentence or "}" in sentence:
            raise ValueError("static annotation contains a runtime format slot")
        encoded = sentence.encode("utf-8")
        if len(encoded) > text_length:
            raise ValueError("external annotation exceeds text_length")
        sentences.append(sentence)
    result = torch.zeros(
        frames.shape[0], text_length, dtype=torch.long, device=frames.device
    )
    for index, sentence in enumerate(sentences):
        raw = torch.tensor(
            [value + 1 for value in sentence.encode("utf-8")],
            dtype=torch.long,
            device=frames.device,
        )
        result[index, : raw.numel()] = raw
    return result
